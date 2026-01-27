#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 14:43:34 2026

@author: m484s199
"""

# %load_ext autoreload
# %autoreload 2

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

hvplot.extension('bokeh')
#%%
# Useful projections
epsg_3413 = ccrs.Stereographic(
    central_latitude=90,  # Tangent at the North Pole
    central_longitude=0,  # Common central meridian
    true_scale_latitude=60 # Latitude where scale is true (or near pole for polar stereo)
)
latlng = ccrs.PlateCarree()
features = gf.ocean.options(scale='50m').opts(projection=epsg_3413) * gf.coastline.options(scale='50m').opts(projection=epsg_3413)
#%%
# Establish an OPR session
# You'll probably want to set a cache directory if you're running this locally to speed
# up subsequent requests. You can do other things like customize the STAC API endpoint,
# but you shouldn't need to do that for most use cases.
opr = xopr.opr_access.OPRConnection(cache_dir="/tmp")

# Or you can open a connection without a cache directory (for example, if you're parallelizing
# this on a cloud cluster without persistent storage).
#opr = xopr.OPRConnection()
#%%
petermann_gdf = gpd.read_file('./petermann_merged_basin_clip.gpkg')
petermann_gdf = petermann_gdf.dissolve()
peter_geom = petermann_gdf.iloc[0].geometry
#%%
peter_latlon = petermann_gdf.to_crs(latlng.proj4_init)
peter_geom_latlon = peter_latlon.iloc[0].geometry
date_range = '2008-01-01T00:00:00Z/2025-01-01T00:00:00Z'

stac_items = opr.query_frames(geometry=peter_geom_latlon, date_range=date_range ,max_items=50)
#%%
# Plot a map of our loaded data over the selected region on an EPSG:3031 projection
# peter_geom_latlon = peter_geom.to_crs(latlng.proj4_init)
# Create a GeoViews object for the selected region
region_gv = gv.Polygons(peter_latlon, crs=latlng).opts(
    color='green',
    line_color='black',
    fill_alpha=0.5,
    projection=epsg_3413,
)
# Plot the frame geometries
frame_lines = []
for item in stac_items.geometry:
    frame_lines.append(gv.Path(item, crs=latlng).opts(
        line_width=2,
        projection=epsg_3413
    ))

(features * region_gv * gv.Overlay(frame_lines)).opts(projection=epsg_3413)
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
    layer_twtt = layer_twtt.reindex(slow_time=radar_ds.slow_time, method='nearest', tolerance=pd.Timedelta(seconds=0.1), fill_value=np.nan)
    
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
#%%
reflectivity = surface_bed_reflection_power(stac_items.loc['Data_20100420_03_009'], opr=opr)
bed_power_grad = np.gradient( np.gradient(reflectivity['bed_power_dB']) ) 
reflectivity['bed_power_grad'] = (('slow_time'), bed_power_grad)

fig, ax = plt.subplots(figsize=(8, 4))
ax2 = ax.twinx()
reflectivity['surface_power_dB'].plot(ax=ax, x='slow_time', label='Surface')
reflectivity['bed_power_dB'].plot(ax=ax, x='slow_time', label='Bed')
reflectivity['bed_power_grad'].plot(ax=ax2, x='slow_time', label='grad',color='tab:green')


ax.set_ylabel('Power [dB]')
ax.legend()
ax2.legend()
plt.show()

