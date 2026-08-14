#!/usr/bin/env python
"""
calib.py -- image->ground homography for the cabbage robot's forward camera, so cross-track can be
reported in centimetres instead of pixels.

The homography is built ANALYTICALLY from the measured mounting of the cabbage robot's camera
(Logitech HD C270 @ 640x480), assuming flat ground and the camera on the robot centreline:
  - height above ground        : 65 cm;
  - optical-axis tilt          : 55 deg from vertical, i.e. 35 deg below the horizontal.
The camera intrinsics K and distortion D are reused from the sibling-thesis (Nguyen Van Son)
checkerboard calibration of the SAME camera unit -- intrinsics depend on the lens, not the mounting,
so reusing them is valid; only the extrinsics (height/tilt) are taken from this robot's install.

This replaces an earlier version that reused a 4-point pixel<->world homography from the sibling robot.
Recovering the extrinsics implied by that borrowed homography gave ~22.6 cm height and ~19 deg below
horizontal -- far from this robot's actual 65 cm / 35 deg-below -- so the borrowed homography
under-reported centimetre cross-track by ~1.7x. The extrinsics here are this robot's measured pose.

World frame: X = lateral (right +), Y = forward (+), Z = up (+), origin on the ground under the camera.

px_to_ground_mm(x,y) -> (X_mm, Y_mm) on the ground plane.
crosstrack_cm(coeffs,H,W) / px_cross_to_cm(ct_px,H,W) -> lateral cross-track in cm.
"""
from __future__ import annotations
import numpy as np
import cv2

import guidance_curve as gc

# Calibration intrinsics are defined at this image resolution (the C270 stream the robot used).
CALIB_W, CALIB_H = 640, 480

# Measured mounting of the cabbage-robot camera.
CAM_HEIGHT_MM = 650.0                    # 65 cm above the ground
CAM_TILT_FROM_VERTICAL_DEG = 55.0        # optical axis 55 deg off straight-down (nadir)
PITCH_BELOW_HORIZON_DEG = 90.0 - CAM_TILT_FROM_VERTICAL_DEG   # = 35 deg below horizontal

# intrinsics / distortion of the same C270 unit (sibling-thesis checkerboard calibration).
K = np.array([[797.4981, 0.0, 323.1862], [0.0, 797.8648, 263.6120], [0.0, 0.0, 1.0]])
DIST = np.array([0.0202, 0.2425, 0.0053, -3.4086e-4, 0.0])  # thesis order [k1, k2, k3, p1, p2]


def _ground_to_img():
    """Pinhole homography mapping ground (X,Y,Z=0) in mm to UNDISTORTED image pixels."""
    p = np.radians(PITCH_BELOW_HORIZON_DEG)
    # world->camera rotation: camera x=right (=world X), optical axis = (0, cos p, -sin p) in world
    # (forward and p below horizontal); no roll, no yaw.
    Rwc = np.array([[1.0, 0.0, 0.0],
                    [0.0, -np.sin(p), -np.cos(p)],
                    [0.0, np.cos(p), -np.sin(p)]])
    t = -Rwc @ np.array([0.0, 0.0, CAM_HEIGHT_MM])     # world origin in camera frame
    return K @ np.column_stack([Rwc[:, 0], Rwc[:, 1], t])


H_GND2IMG = _ground_to_img()
H_IMG2GND = np.linalg.inv(H_GND2IMG)     # UNDISTORTED image px -> ground mm


def px_to_ground_mm(x, y):
    # undistort the raw pixel into the pinhole frame, then map to the ground plane
    u = cv2.undistortPoints(np.array([[[float(x), float(y)]]], np.float32), K, DIST, P=K)
    xu, yu = float(u[0, 0, 0]), float(u[0, 0, 1])
    g = H_IMG2GND @ np.array([xu, yu, 1.0])
    return g[0] / g[2], g[1] / g[2]


def px_cross_to_cm(ct_px, H, W):
    """Convert a pixel cross-track (x_row - W/2 at the near row) to cm on the ground."""
    yn = gc.Y_CROSSTRACK * H
    xr, _ = px_to_ground_mm(W / 2.0 + ct_px, yn)
    xc, _ = px_to_ground_mm(W / 2.0, yn)
    return (xr - xc) / 10.0


def crosstrack_cm(coeffs, H, W):
    yn = gc.Y_CROSSTRACK * H
    xr, _ = px_to_ground_mm(gc.x_at(coeffs, yn), yn)
    xc, _ = px_to_ground_mm(W / 2.0, yn)
    return (xr - xc) / 10.0


# Virtual safety corridor (platform thesis): the straddle robot spans one bed, so the central row lies
# within the chassis safe width Wsafe=440 mm (+/-220 mm) of the centreline; the two neighbour rows fall
# outside it. Projecting that ground corridor to image pixels per row gives a perspective-correct band that
# excludes the side rows -- the fix for wrong-row lock-on.
WSAFE_MM = 440.0


def corridor_px(y, W, half_mm=WSAFE_MM / 2.0):
    """Image-pixel x-bounds (x_lo, x_hi) of the +/-half_mm ground corridor around the robot centreline at
    image row y, or None if degenerate. Wide near the robot, narrow far away (perspective)."""
    x0, _ = px_to_ground_mm(W / 2.0, y)
    x1, _ = px_to_ground_mm(W / 2.0 + 1.0, y)
    mm_per_px = x1 - x0
    if abs(mm_per_px) < 1e-6:
        return None
    x_center = W / 2.0 - x0 / mm_per_px            # image column where ground X = 0 (the centreline)
    half_px = abs(half_mm / mm_per_px)
    return x_center - half_px, x_center + half_px


if __name__ == "__main__":
    print(f"install: height {CAM_HEIGHT_MM/10:.0f} cm, tilt {CAM_TILT_FROM_VERTICAL_DEG:.0f} deg "
          f"from vertical (= {PITCH_BELOW_HORIZON_DEG:.0f} deg below horizontal)")
    yn = gc.Y_CROSSTRACK * CALIB_H
    _, depth = px_to_ground_mm(CALIB_W / 2.0, yn)
    print(f"near cross-track row y={yn:.0f}px -> ground depth {depth/10:.0f} cm ahead")
    print(f"1 px lateral at near row ~= {abs(px_cross_to_cm(1.0, CALIB_H, CALIB_W)):.4f} cm")
    print(f"100 px cross-track ~= {px_cross_to_cm(100.0, CALIB_H, CALIB_W):.2f} cm")
