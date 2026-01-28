#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 12:15:26 2026

@author: m484s199
"""

import numpy as np
from scipy.special import erf
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def erf_topography_model(x, a, b, x0, c):
    """
    S-shaped model for topographic transitions.
    a: half-height of the step
    b: transition steepness (1/width)
    x0: center of the transition
    c: vertical offset
    """
    return a * erf(b * (x - x0)) + c

def get_derivatives(x, a, b, x0):
    """
    Calculates the 1st, 2nd, and 3rd analytical derivatives of the fit.
    """
    u = b * (x - x0)
    # 1st derivative: Gaussian
    z_prime = (2 * a * b / np.sqrt(np.pi)) * np.exp(-u**2)
    # 2nd derivative
    z_double_prime = -(4 * a * b**2 / np.sqrt(np.pi)) * u * np.exp(-u**2)
    # 3rd derivative
    z_triple_prime = (4 * a * b**3 / np.sqrt(np.pi)) * (2 * u**2 - 1) * np.exp(-u**2)
    
    return z_prime, z_double_prime, z_triple_prime

# --- Example Usage ---

# 1. Generate synthetic grounding zone topography
x_data = np.linspace(0, 50, 500)
# Grounded ice (high) to floating ice (low)
y_true = -50 * erf(0.15 * (x_data - 25)) - 300 
y_noise = y_true + np.random.normal(0, 1.5, x_data.size)

# 2. Fit the model to the data
popt, _ = curve_fit(erf_topography_model, x_data, y_noise, p0=[-50, 0.1, 25, -300])
a_fit, b_fit, x0_fit, c_fit = popt

# 3. Find the "Point of Maximum Curvature" (Root of 3rd derivative)
# For erf(u), roots of z''' are at u = +/- 1/sqrt(2)
x_knee_1 = x0_fit + (1 / (b_fit * np.sqrt(2)))
x_knee_2 = x0_fit - (1 / (b_fit * np.sqrt(2)))

# 4. Visualization
y_fit = erf_topography_model(x_data, *popt)
_, _, z3 = get_derivatives(x_data, a_fit, b_fit, x0_fit)

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.scatter(x_data, y_noise, s=2, color='gray', alpha=0.5, label='Radar Data')
plt.plot(x_data, y_fit, 'r', label='ERF Fit')
plt.axvline(x_knee_1, color='blue', linestyle='--', label='Max Curvature Point')
plt.title("Basal Topography Fit")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(x_data, z3, 'g', label='3rd Derivative')
plt.axhline(0, color='black', lw=0.5)
plt.axvline(x_knee_1, color='blue', linestyle='--')
plt.title("Third Derivative (identifying the break)")
plt.tight_layout()
plt.show()