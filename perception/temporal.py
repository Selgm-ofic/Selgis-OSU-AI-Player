"""Pseudo-three-channel vision input for the fixed monochrome osu! skin."""
import cv2
import numpy as np


CHANNELS = ("gray_current", "absdiff_x3", "canny_edges")


def to_gray(frame):
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def temporal_frame(frame, previous_gray=None):
    """Pack intensity, short-term motion and contours into uint8 HxWx3."""
    gray = to_gray(frame)
    if previous_gray is None or previous_gray.shape != gray.shape:
        difference = np.zeros_like(gray)
    else:
        difference = cv2.absdiff(gray, previous_gray)
        difference = cv2.convertScaleAbs(difference, alpha=3.0)
    edges = cv2.Canny(gray, 45, 125, L2gradient=True)
    # A one-pixel edge is fragile after resizing in YOLO; retain the contour
    # while keeping neighbouring circles separable.
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    return cv2.merge((gray, difference, edges)), gray
