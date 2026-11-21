import numpy as np
from .config import deriv

HEAD_RADIUS_MM = 50.0  # converts rotations (radians) to mm of motion at the cortex (Power et al. 2012)


def params(run):
    """Rigid-body parameters per volume: tx ty tz (mm), rx ry rz (rad), relative to the run's reference."""
    return np.loadtxt(deriv(f'run{run}_motion.txt'))


def framewise_displacement(p):
    d = np.abs(np.diff(p, axis=0))
    return np.r_[0.0, d[:, :3].sum(1) + HEAD_RADIUS_MM * d[:, 3:].sum(1)]


def confounds24(p):
    """Friston-24: params, their derivatives, and squares of both."""
    dp = np.vstack([np.zeros(6), np.diff(p, axis=0)])
    return np.hstack([p, dp, p ** 2, dp ** 2])
