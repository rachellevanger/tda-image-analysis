# TDA Image Analysis Project : core.standard_analysis
#
# Copyright (C) 2016-2017 TDA Image Analysis Project
# Authors: Shaun Harker <sharker81@gmail.com>
#          Rachel Levanger <rachel.levanger@gmail.com>

"""
Standard numerical techniques for working with image data.
"""

import scipy.ndimage
import numpy as np
import math
import pandas as pd


def orientation_field(bmp, sigma=3):
    # Author: Shaun Harker, 2016
    # Based on algorithm by Bazen and Gerez from "Systematic methods for the 
    # computation of the directional fields and singular points of fingerprints," 2002.
    """
    Computes orientation field (result everywhere between -pi/2 and pi/2)
    from the given vector field.
    """
    u = bmp.astype(float)
    du = np.gradient(u)
    [ux, uy] = du
    Y = scipy.ndimage.filters.gaussian_filter(2.0*ux*uy, sigma=sigma)
    X = scipy.ndimage.filters.gaussian_filter(ux**2.0 - uy**2.0, sigma=sigma)
    return .5 * np.arctan2(Y, X)

def topological_defect_array(orientation_field):
    """
    Returns a matrix of topological defects for the given orientation field. 
    Each entry in the matrix is the charge of the defect.
    """
    JX = np.diff(orientation_field, axis=0)
    JY = np.diff(orientation_field, axis=1)
    JX += math.pi * (JX < -math.pi/2.0 ) - math.pi * (JX > math.pi/2.0)
    JY += math.pi * (JY < -math.pi/2.0 ) - math.pi * (JY > math.pi/2.0)
    return np.rint((np.diff(JY, axis=0) - np.diff(JX, axis=1))/math.pi)

def topological_defect_array_to_dataframe(td_array):
    """
    Converts an array of topological defects to a dataframe containing 
    a list of defect locations and their charges.
    """
    td = np.asarray([ (i,j,td_array[i,j]) for (i,j) in np.argwhere(td_array)])
    td = td.astype(np.int)
    td = pd.DataFrame(td, columns=['row', 'col', 'type'])

    return td

def fourier_diff(u, order=1):
    [N, M] = u.shape
    [kx, ky] = np.mgrid[0:N,0:M]
    kx = kx - float(N) * ( kx > float(N)/2.0 )
    ky = ky - float(M) * ( ky > float(M)/2.0 )    
    if order % 2 == 1 and N % 2 == 0: kx[N//2,:] = 0.0
    if order % 2 == 1 and M % 2 == 0: ky[:,M//2] = 0.0
    kx = (kx * 2.0 * math.pi * 1j / float(N)) ** order
    ky = (ky * 2.0 * math.pi * 1j / float(M)) ** order
    u_fft = np.fft.fft2(u)
    ux_fft = u_fft * kx
    uy_fft = u_fft * ky
    ux = np.real(np.fft.ifft2(ux_fft))
    uy = np.real(np.fft.ifft2(uy_fft))
    return [ux, uy]

def emb_wavenumber(u, method="difference", smoothing=10):
    # Author: Shaun Harker, 2016
    # Implementation of the EMB algorithm, from "A new fast method for determining local 
    # properties of striped patterns.," 1997.

    u = scipy.ndimage.filters.gaussian_filter(u, sigma=2)
    u = u - np.sum(u)/np.size(u)
    u = u / np.max(np.absolute(u))
    if method == "difference":
        [ux, uy] = np.gradient(u)
        [uxx, uxy] = np.gradient(ux)
        [uxy, uyy] = np.gradient(uy)
        uxxx = np.gradient(uxx)[0]
        uyyy = np.gradient(uyy)[1]
    elif method == "fourier":
        [ux, uy] = fourier_diff(u, order=1)
        [uxx, uyy] = fourier_diff(u, order=2)
        [uxxx, uyyy] = fourier_diff(u, order=3)
        uxy = fourier_diff(ux)[1]
    else:
        raise ValueError('EMB_Wavenumber: Unrecognized method "' + method + '"')
    Test1 = np.absolute(u) > np.maximum(np.absolute(ux),np.absolute(uy))
    Test2 = np.absolute(uxx) > np.absolute(uyy)
    Test3 = np.absolute(ux) > np.absolute(uy)
    Case1 = np.logical_and(Test1, Test2)
    Case2 = np.logical_and(Test1, np.logical_not(Test2))
    Case3 = np.logical_and(np.logical_not(Test1), Test3)
    Case4 = np.logical_and(np.logical_not(Test1), np.logical_not(Test3))
    Case1kx = np.nan_to_num(np.sqrt(np.absolute(uxx / u)))
    Case1ky = Case1kx * np.nan_to_num(uxy / uxx )
    Case2ky = np.nan_to_num(np.sqrt(np.absolute(uyy / u )))
    Case2kx = Case2ky * np.nan_to_num(uxy / uyy)
    Case3kx = np.nan_to_num(np.sqrt(np.absolute(uxxx / ux)))
    Case3ky = Case3kx * np.nan_to_num(uy / ux )
    Case4ky = np.nan_to_num(np.sqrt(np.absolute(uyyy / uy)))
    Case4kx = Case4ky * np.nan_to_num(ux / uy)
    kx = Case1 * Case1kx + Case2 * Case2kx + Case3 * Case3kx + Case4 * Case4kx
    ky = Case1 * Case1ky + Case2 * Case2ky + Case3 * Case3ky + Case4 * Case4ky
    Sign = 1.0 - 2.0 * (kx < 0)
    kx = kx * Sign
    ky = ky * Sign
    wn = np.sqrt(kx**2.0 + ky**2.0)

    # Added by: Rachel Levanger, 2017
    # Set NaN values to median non-NaN wavenumber value and then smooth
    wn[np.isnan(wn)] = np.median(wn[np.logical_not(np.isnan(wn))])
    wn = scipy.ndimage.filters.gaussian_filter(wn, sigma=smoothing)

    return wn

