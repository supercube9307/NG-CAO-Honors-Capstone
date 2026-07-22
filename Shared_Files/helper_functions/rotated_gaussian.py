"""
Python conversion of rotated_gaussian.m

https://claude.ai/chat/821bf681-6e20-4bc5-a676-f6cc65fd18ae

Original MATLAB signature:
    function [y, fitted_gaussian] = rotated_gaussian(par)

par = [offset, amplitude, theta, x0, y0, sigmax, sigmay]

Because scipy's least_squares (unlike the custom LMFnlsq used in MATLAB)
does not support MATLAB-style global variables, frame_to_fit / xp / yp /
ifit are passed in explicitly as arguments instead of being read from
globals. A module-level counter replaces the MATLAB global `iter` so the
same per-evaluation progress message can still be printed.
"""

import itertools
import numpy as np

# Mirrors the MATLAB "global iter" counter, incremented once per function call
_iter_counter = itertools.count(1)


def rotated_gaussian(par, xp, yp, frame_to_fit, ifit=1, verbose=True):
    """
    Evaluate the rotated-2D-Gaussian model and its residual against
    frame_to_fit, exactly mirroring the MATLAB function of the same name.

    Parameters
    ----------
    par : array_like, shape (7,)
        [offset, amplitude, theta, x0, y0, sigmax, sigmay]
    xp, yp : ndarray
        Meshgrid coordinate arrays (same shape as frame_to_fit).
    frame_to_fit : ndarray
        The image subframe being fit.
    ifit : float
        Same role as the MATLAB global `ifit` (usually 1).
    verbose : bool
        If True, print progress like the original `disp(...)` call.

    Returns
    -------
    y : ndarray, shape (N,)
        Flattened residual (fit_function - ifit*frame_to_fit), the
        quantity scipy.optimize.least_squares needs.
    fitted_gaussian : ndarray
        The 2D fitted model, same shape as xp/yp.
    """
    offset, amplitude, theta, x0, y0, sigmax, sigmay = par

    fitted_gaussian = offset + amplitude * np.exp(
        -(np.cos(theta) * (xp - x0) - np.sin(theta) * (yp - y0)) ** 2 / (2 * sigmax ** 2)
        - (np.sin(theta) * (xp - x0) + np.cos(theta) * (yp - y0)) ** 2 / (2 * sigmay ** 2)
    )

    fit_eval = fitted_gaussian - ifit * frame_to_fit
    y = fit_eval.ravel(order='F')  # MATLAB (:) flattens column-major

    if verbose:
        n = next(_iter_counter)
        print(f"rotated_gaussian:  iter={n}   {np.linalg.norm(y)}")

    return y, fitted_gaussian


def rotated_gaussian_residuals(par, xp, yp, frame_to_fit, ifit=1, verbose=True):
    """
    Thin wrapper returning ONLY the residual vector, in the form
    scipy.optimize.least_squares expects for its `fun` argument
    (it cannot accept a function that returns a tuple).
    """
    y, _ = rotated_gaussian(par, xp, yp, frame_to_fit, ifit=ifit, verbose=verbose)
    return y
