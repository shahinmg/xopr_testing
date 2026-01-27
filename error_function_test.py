#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 17:17:10 2026

@author: laserglaciers
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import curve_fit

# 1. Define the Error Function model
def erf_fit_func(x, A, x0, sigma, B):
    """
    A: Amplitude (half the elevation change)
    x0: Center of transition (Grounding Line estimate)
    sigma: Width of transition
    B: Vertical offset
    """
    return A * erf((x - x0) / sigma) + B

# 2. Example Data (Simulating an ice-bed elevation profile)
# x is distance along the survey line, y is elevation
x_data = np.linspace(0, 50, 100)
# Create a synthetic S-curve with some noise
y_true = 50 * erf((x_data - 25) / 5) - 200 
y_data = y_true + np.random.normal(0, 2, x_data.size)

# 3. Fit the function to the data
# Initial guesses: [Amp=50, center=25, width=5, offset=-200]
initial_guess = [50, 25, 5, -200]
params, covariance = curve_fit(erf_fit_func, x_data, y_data, p0=initial_guess)

A_fit, x0_fit, sigma_fit, B_fit = params
print(f"Estimated Grounding Center (x0): {x0_fit:.2f}")

# 4. Generate the fit curve for plotting
y_fit = erf_fit_func(x_data, *params)

# Plotting
plt.scatter(x_data, y_data, label='Radar Data (Smoothed)', color='black', s=10)
plt.plot(x_data, y_fit, 'g--', label='Error Function Fit', linewidth=2)
plt.axvline(x0_fit, color='red', linestyle=':', label='Grounding Line')
plt.xlabel('Distance (km)')
plt.ylabel('Bed Elevation (m)')
plt.legend()
plt.show()