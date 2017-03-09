# -*- mode: python; coding: utf-8 -*-
# Copyright 2017 Peter Williams and collaborators
# Licensed under the MIT license.

"""Polarized radiative transfer with Jason Dexter's grtrans code.

The canonical ordering of the key physical parameters is set to match Symphony
and my large computations: `(nu, B, n_e, theta, p)`.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from pwkit.numutil import broadcastize

from .polsynchemis import polsynchemis
from .radtrans_integrate import radtrans_integrate


# Different integration methods. The demo uses LSODA. I needed to use FORMAL
# in order to be able to trace rays up to (realistic) "x" ~ 1e10; other
# methods only worked with max(x) ~ 1e5.

METHOD_LSODA_YES_LINEAR_STOKES = 0 # LSODA with IS_LINEAR_STOKES=1
METHOD_DELO = 1 # DELO method from Rees+ (1989ApJ...339.1093R)
METHOD_FORMAL = 2 # "formal" method; may be "matricant (O-matrix) method from Landi Degl'Innocenti"?
METHOD_LSODA_NO_LINEAR_STOKES = 3 # LSODA with IS_LINEAR_STOKES=0


def integrate_ray (x, j, K, atol=1e-8, rtol=1e-6, method=METHOD_FORMAL, tau_max=10.):
    """Arguments:

    x
      1D array, shape (n,). "path length along the ray starting from its minimum"
    j
      Array, shape (n, 4). Emission coefficients.
    K
      Array, shape (n, 7). Absorption coefficients and Faraday mixing coefficients:
     (alpha_{IQUV}, rho_{123}).
    atol
      Some kind of tolerance parameter.
    rtol
      Some kind of tolerance parameter.
    method
      The integration method to use; see METHOD_* constants.
    tau_max
      The maximum optical depth for the LSODA methods.

    Returns: Array of shape (4, n): Stokes intensities along the ray.

    """
    n = x.size
    from scipy.integrate import cumtrapz

    radtrans_integrate.init_radtrans_integrate_data (
        method, # method selector
        4, # number of equations
        n, # number of input data points
        n, # number of output data points
        tau_max, # maximum optical depth
        0.1, # maximum absolute step size
        atol, # absolute tolerance
        rtol, # relative tolerance
        1e-2, # "thin" parameter for DELO method ... to be researched
        100000, # maximum number of steps
    )

    tau = np.append (0., cumtrapz (K[:,0], x)) # shape (n,)
    radtrans_integrate.integrate (x[::-1], j, K, tau, 4)
    i = radtrans_integrate.intensity.copy ()
    radtrans_integrate.del_radtrans_integrate_data ()
    return i


@broadcastize(7, ret_spec=None)
def calc_powerlaw_synchrotron_coefficients (nu, B, n_e, theta, p, gamma_min, gamma_max):
    """Jason Dexter writes: "polsynchpl is only very accurate for p = 3, 3.5, 7
    because it uses numerically tabulated integrals. For other values of p it
    interpolates or extrapolates."

    Returns an array of shape (X, 11), where X is the input shape and:

    - `array[:,:4]` are the emission coefficients.
    - `array[:,4:8]` are the absorption coefficients.
    - `array[:,8:]` are the Faraday mixing coefficients.

    NOTE: in the past I had found cases with unphysical emission coefficients
    where j_{Q,U,V}**2 added up to be larger than j_I**2. Hasn't been
    reverified for a little while, though.

    NOTE: at some point I talked myself into believing that I couldn't get
    grtrans and Symphony to agree for gamma_min >~ 0.5; I needed gamma_min 0.1
    to get agreement. But now I find the opposite! Sigh.

    """
    assert nu.ndim == 1

    # segfault if size = 1, but everything seems to work OK if we just lie ...
    size = max(nu.size, 2)

    polsynchemis.initialize_polsynchpl (size)
    chunk = polsynchemis.polsynchpl (nu, n_e, B, theta, p, gamma_min, gamma_max)
    polsynchemis.del_polsynchpl (size)
    return chunk


# Slightly higher-level wrappers ...

@broadcastize(7, ret_spec=None)
def calc_powerlaw_nontrivial (nu, B, n_e, theta, p, gamma_min, gamma_max):
    """Like `calc_powerlaw_synchrotron_coefficients`, but packs the outputs in the
    same way as used by my work with Symphony.

    """
    chunk = calc_powerlaw_synchrotron_coefficients(nu, B, n_e, theta, p, gamma_min, gamma_max)

    result = np.empty(nu.shape + (6,))
    result[...,0] = chunk[...,0]
    result[...,2] = chunk[...,1]
    result[...,4] = chunk[...,3]
    result[...,1] = chunk[...,4]
    result[...,3] = chunk[...,5]
    result[...,5] = chunk[...,7]

    return result
