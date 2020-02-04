import cv2
import numpy as np

width = 600
height = 600


def fractal(pmin, pmax, qmin, qmax,
            max_iterations=122, infinity_border=10):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    p, q = np.mgrid[pmin:pmax:(width * 1j), qmin:qmax:(height * 1j)]

    c = p + 1j * q
    z = np.zeros_like(c)

    for k in range(max_iterations):
        z = z ** 2 + c

        mask = (np.abs(z) > infinity_border) & (np.add.reduce(canvas, axis=2) == 0)

        canvas[mask] = (k * 2, 150, 255)
        z[mask] = np.nan
    return canvas


max_frames = 300
max_zoom = 200
pmin, pmax, qmin, qmax = -2.5, 1.5, -2, 2


def render_frame(i):
    print(f'\r{i + 1}/{max_frames}', end='')

    p_center, q_center = -0.793191078177363, 0.16093721735804
    zoom = (i / max_frames * 2) ** 3 * max_zoom + 1
    scalefactor = 1 / zoom

    pmin_ = (pmin - p_center) * scalefactor + p_center
    qmin_ = (qmin - q_center) * scalefactor + q_center
    pmax_ = (pmax - p_center) * scalefactor + p_center
    qmax_ = (qmax - q_center) * scalefactor + q_center

    frame = fractal(pmin_, pmax_, qmin_, qmax_)
    return cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)


frames = [render_frame(i) for i in range(max_frames)]
while True:
    for f in frames:
        cv2.imshow('fractal', f)
        cv2.waitKey(2)
    for f in reversed(frames):
        cv2.imshow('fractal', f)
        cv2.waitKey(2)
