# -*- coding: utf-8 -*-
"""
Created on Mon Mar 2 17:48:13 2026

@author: unsal
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


d = 3e-3  

D1_b = 5.9475
D2_b = 13.3846

I_b = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])

sigma_minus_b = np.array([13.1630, 13.0147, 13.1506, 13.0317, 12.9320, 12.7188, 12.6990, 12.3853])
sigma_plus_b = np.array([13.5683, 13.3578, 13.7705, 13.8426, 13.9532, 14.3175, 14.2300, 14.1262])


D1_r = 12.6823
D2_r = 18.6472

I_r = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])

sigma_minus_r = np.array([12.5662, 12.4925, 12.5124, 12.3798, 12.3482, 12.2681, 12.2488, 12.2346])
sigma_plus_r = np.array([13.0001, 13.0603, 13.0722, 12.9948, 13.2568, 13.2817, 13.4075, 13.2916])


def deltanu(I, D1, D2, sigma_minus, sigma_plus):
    B = 0.058 * I
    delta_nu = (1/(2*d)) * ((sigma_plus**2 - sigma_minus**2)/(D2**2 - D1**2))
    delta_nu = np.abs(delta_nu)
    return B, delta_nu


coeff = 46.686

gu_b = 2
Mu_b_plus = 1
Mu_b_minus = -1
gl_b = 0
Ml_b = 0

gu_r = 1
Mu_r_plus = 1
Mu_r_minus = -1
gl_r = 1
Ml_r = 0


def theory(gu, Mu, gl, Ml):
    return coeff * (gu*Mu - gl*Ml)


B_b, delta_nu_b = deltanu(I_b, D1_b, D2_b, sigma_minus_b, sigma_plus_b)
B_r, delta_nu_r = deltanu(I_r, D1_r, D2_r, sigma_minus_r, sigma_plus_r)

slope_b, C_b, r_b, _, err_b = stats.linregress(B_b, delta_nu_b)
slope_r, C_r, r_r, _, err_r = stats.linregress(B_r, delta_nu_r)

fit_b = slope_b * B_b + C_b
fit_r = slope_r * B_r + C_r


slope_th_b = theory(gu_b, Mu_b_plus, gl_b, Ml_b) - theory(gu_b, Mu_b_minus, gl_b, Ml_b)
slope_th_r = theory(gu_r, Mu_r_plus, gl_r, Ml_r) - theory(gu_r, Mu_r_minus, gl_r, Ml_r)

theory_b = slope_th_b * B_b
theory_r = slope_th_r * B_r


print("BLUE")
print("Slope:", slope_b)
print("Slope TH:", slope_th_b)
print("Intercept:", C_b)
print("R^2:", r_b**2)
print(" ")

print("RED")
print("Slope:", slope_r)
print("Slope TH:", slope_th_r)
print("Intercept:", C_r)
print("R^2:", r_r**2)
print(" ")

print("Experimental slope ratio (Red / Blue):", slope_r / slope_b)

print("Theoretical slope ratio (Red / Blue):", slope_th_r / slope_th_b)


plt.figure()

plt.scatter(B_b, delta_nu_b)
plt.plot(B_b, fit_b, label="Blue light")

plt.scatter(B_r, delta_nu_r)
plt.plot(B_r, fit_r, label="Red light")

plt.plot(B_b, theory_b, '--', label="Blue theory")
plt.plot(B_r, theory_r, '--', label="Red theory")

plt.xlabel("Magnetic Field B (T)")
plt.ylabel("Zeeman Splitting Δv (m^-1)")
plt.title("Zeeman Splitting vs Magnetic Field (Red & Blue)")
plt.legend()

plt.show()