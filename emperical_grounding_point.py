#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 13:24:22 2026

@author: m484s199
"""

import numpy as np
import xarray as xr
import hvplot.xarray
import geoviews as gv
import geoviews.feature as gf
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import shapely
import scipy.constants
import pandas as pd
import traceback
import geopandas as gpd
import xopr.opr_access
import xopr.geometry
import dask
from dask.distributed import LocalCluster
import dask
import dask.delayed as delayed
from dask import compute
from dask.distributed import LocalCluster
import time
import requests

#%%


# Useful projections
epsg_3413 = ccrs.Stereographic(
    central_latitude=90,  # Tangent at the North Pole
    central_longitude=0,  # Common central meridian
    true_scale_latitude=60 # Latitude where scale is true (or near pole for polar stereo)
)
latlng = ccrs.PlateCarree()
features = gf.ocean.options(scale='50m').opts(projection=epsg_3413) * gf.coastline.options(scale='50m').opts(projection=epsg_3413)


# Establish an OPR session
# You'll probably want to set a cache directory if you're running this locally to speed
# up subsequent requests. You can do other things like customize the STAC API endpoint,
# but you shouldn't need to do that for most use cases.
opr = xopr.opr_access.OPRConnection(cache_dir="/tmp")

# Or you can open a connection without a cache directory (for example, if you're parallelizing
# this on a cloud cluster without persistent storage).
#opr = xopr.OPRConnection()

petermann_gdf = gpd.read_file('./petermann_merged_basin_clip.gpkg')
petermann_gdf = petermann_gdf.dissolve()
peter_geom = petermann_gdf.iloc[0].geometry

peter_latlon = petermann_gdf.to_crs(latlng.proj4_init)
region = peter_latlon.iloc[0].geometry

xopr.geometry.project_geojson(region)

date_range = '2008-01-01T00:00:00Z/2025-01-01T00:00:00Z'
stac_items = opr.query_frames(geometry=region, date_range=date_range ,max_items=50)

#%%

def extract_layer_peak_power(radar_ds, layer_twtt, margin_twtt):
    """
    Extract the peak power of a radar layer within a specified margin around the layer's two-way travel time (TWTT).

    Parameters:
    - radar_ds: xarray Dataset containing radar data.
    - layer_twtt: The two-way travel time of the layer to extract.
    - margin_twtt: The margin around the layer's TWTT to consider for peak power extraction.

    Returns:
    - A DataArray containing the peak power values for the specified layer.
    """
    
    # Ensure that layer_twtt.slow_time matches the radar_ds slow_time
    t_start = np.minimum(radar_ds.slow_time.min(), layer_twtt.slow_time.min())
    t_end = np.maximum(radar_ds.slow_time.max(), layer_twtt.slow_time.max())
    layer_twtt = layer_twtt.sel(slow_time=slice(t_start, t_end))
    radar_ds = radar_ds.sel(slow_time=slice(t_start, t_end))
    #layer_twtt = layer_twtt.interp(slow_time=radar_ds.slow_time, method='nearest')
    layer_twtt = layer_twtt.reindex(slow_time=radar_ds.slow_time, method='nearest', tolerance=pd.Timedelta(seconds=1), fill_value=np.nan)
    
    # Calculate the start and end TWTT for the margin
    start_twtt = layer_twtt - margin_twtt
    end_twtt = layer_twtt + margin_twtt
    
    # Extract the data within the specified TWTT range
    data_within_margin = radar_ds.where((radar_ds.twtt >= start_twtt) & (radar_ds.twtt <= end_twtt), drop=True)

    power_dB = 10 * np.log10(np.abs(data_within_margin.Data))

    # Find the twtt index corresponding to the peak power
    peak_twtt_index = power_dB.argmax(dim='twtt')
    # Convert the index to the actual TWTT value
    peak_twtt = power_dB.twtt[peak_twtt_index]

    # Calculate the peak power in dB
    peak_power = power_dB.isel(twtt=peak_twtt_index)

    # Remove unnecessary dimensions
    peak_twtt = peak_twtt.drop_vars('twtt')
    peak_power = peak_power.drop_vars('twtt')
    
    return peak_twtt, peak_power

def surface_bed_reflection_power(stac_item, opr=xopr.opr_access.OPRConnection()):

    frame = opr.load_frame(stac_item, data_product='CSARP_standard')
    frame = frame.resample(slow_time='5s').mean()

    layers = opr.get_layers(frame, source='auto', include_geometry=False)
    if layers is None:
        return None
    
    # Re-pick surface and bed layers to ensure we're getting the peaks
    speed_of_light_in_ice = scipy.constants.c / np.sqrt(3.17)  # Speed of light in ice (m/s)
    layer_selection_margin_twtt = 50 / speed_of_light_in_ice # approx 50 m margin in ice
    surface_repicked_twtt, surface_power = extract_layer_peak_power(frame, layers["standard:surface"]['twtt'], layer_selection_margin_twtt)
    bed_repicked_twtt, bed_power = extract_layer_peak_power(frame, layers["standard:bottom"]['twtt'], layer_selection_margin_twtt)

    # Create a dataset from surface_repicked_twtt, bed_repicked_twtt, surface_power, and bed_power

    reflectivity_dataset = xr.merge([
        surface_repicked_twtt.rename('surface_twtt'),
        bed_repicked_twtt.rename('bed_twtt'),
        surface_power.rename('surface_power_dB'),
        bed_power.rename('bed_power_dB'),
        ],
        compat='override')

    flight_line_metadata = frame.drop_vars(['Data', 'Surface'])
    reflectivity_dataset = xr.merge([reflectivity_dataset, flight_line_metadata])

    reflectivity_dataset = reflectivity_dataset.drop_dims(['twtt'])  # Remove the twtt dimension since everything has been flattened

    attributes_to_copy = ['season', 'segment', 'doi', 'ror', 'funder_text']
    reflectivity_dataset.attrs = {attr: frame.attrs[attr] for attr in attributes_to_copy if attr in frame.attrs}

    return reflectivity_dataset


@delayed
def safe_get_layers_db(stac_item, opr=xopr.opr_access.OPRConnection()):
    try:
        retries = 1
        backoff_time = 5
        backoff_jitter = 30
        while retries > 0:
            try:
                return opr.get_layers_db(stac_item)
            except requests.exceptions.RequestException as e:
                sleep_time = backoff_time + np.random.uniform(0, backoff_jitter)
                print(f"Request error fetching layers for {stac_item['id']}: {e}. Retrying in {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
                retries -= 1
                backoff_time *= 2  # Exponential backoff
    except Exception as e:
        print(f"Error fetching layers for {stac_item['id']}: {e}")
        return None

def get_basal_layer_wgs84(stac_item, preloaded_layer=None, opr=xopr.opr_access.OPRConnection()):
    if (preloaded_layer is None) or len(preloaded_layer) < 2:
        layers = opr.get_layers_files(stac_item)
    else:
        layers = preloaded_layer
    
    basal_layer = layers["standard:bottom"]
    surface_layer = layers["standard:surface"]

    surface_wgs84 = layers["standard:surface"]['elev'] - (layers["standard:surface"]['twtt'] * (scipy.constants.c / 2))
    delta_twtt = basal_layer['twtt'] - surface_layer['twtt']
    basal_wgs84 = surface_wgs84 - (delta_twtt * ((scipy.constants.c / np.sqrt(3.15)) / 2))

    basal_layer['wgs84'] = basal_wgs84
    return basal_layer


#%%

#frames from manually looking on OPS
ops_frames = ['20100420_02_007', '20100420_03_009'] # just to have stored somewhere

petermann_item = stac_items.loc['Data_20100420_03_009']


layers = opr.get_layers_files(petermann_item)

for layer_idx in layers:
    layers[layer_idx] = xopr.radar_util.add_along_track(layers[layer_idx])
    layers[layer_idx] = xopr.layer_twtt_to_range(layers[layer_idx], layers["standard:surface"], vertical_coordinate='wgs84')
    layers[layer_idx] = xopr.layer_twtt_to_range(layers[layer_idx], layers["standard:surface"], vertical_coordinate='range')

#%%
frame_1 = opr.load_frame(petermann_item)
frame_1 = xopr.radar_util.add_along_track(frame_1)
frame_1 = xopr.radar_util.interpolate_to_vertical_grid(frame_1, vertical_coordinate='wgs84')

#%%
clb_min_pct, clb_max_pct = 30, 97

# Plot radargrams in elevation coordinates with layers
fig, (ax1) = plt.subplots(1, 1, figsize=(15, 8))

# Frame 1 radargram in elevation
pwr_1_elev = 10*np.log10(np.abs(frame_1.Data))
vmax_1 = np.percentile(pwr_1_elev, clb_max_pct)
vmin_1 = np.percentile(pwr_1_elev, clb_min_pct)
pwr_1_elev.plot.imshow(x='along_track', y='wgs84', cmap='gray', ax=ax1, vmin=vmin_1, vmax=vmax_1)
# ax1.axvline(frame_1.along_track[idx_1].values, color='red', linestyle='--', linewidth=2, label='Crossover')

# Plot layers using elevation data
for layer_name in layers:
    layers[layer_name]['wgs84'].plot(ax=ax1, x='along_track', linewidth=1, linestyle=':', label=layer_name)

# ax1.set_title(f"{intersect['collection_1']} - {intersect['id_1']} (Elevation view)")
ax1.set_ylabel('Elevation (m)')
ax1.legend()

#%%

reflectivity = surface_bed_reflection_power(petermann_item, opr=opr)
layers['standard:surface']['surface_power_dB'] = reflectivity['surface_power_dB']
layers['standard:bottom']['bed_power_dB'] = reflectivity['bed_power_dB']

#%%

# Plot layers using elevation data
fig2, (ax2, ax3) = plt.subplots(nrows=2, ncols=1, figsize=(8,6))
for layer_name in layers:
    layers[layer_name]['wgs84'].plot(ax=ax2, x='slow_time', linewidth=1, linestyle=':', label=layer_name)

# Plot layers using elevation data
#ax2 plot
reflectivity = surface_bed_reflection_power(stac_items.loc['Data_20100420_03_009'], opr=opr)
bed_power_grad = np.gradient(reflectivity['bed_power_dB'])
reflectivity['bed_power_grad'] = (('slow_time'), bed_power_grad)
ax2.set_title('Petermann Grounding Point Example')


#ax3 plot
reflectivity['bed_power_grad'].plot(ax=ax3, x='slow_time', label='bed_grad', color='tab:green')
grad_max_idx = reflectivity['bed_power_grad'].argmax(dim="slow_time").data
grad_slowtime = reflectivity['bed_power_grad']['slow_time'][grad_max_idx]
bed_point = layers['standard:bottom']['wgs84'].sel(slow_time=grad_slowtime.data, method='nearest')

ax2.scatter(bed_point['slow_time'], bed_point.data, color='r', s=20, label="Grounding Point")

ax3.set_ylabel('Power [dB]')
ax2.legend()
ax3.legend()
ax3.set_title('')

fig2.savefig('auto_grounding_point_example.png')


