import cv2
import numpy as np
from math import cos, sin, radians


# noinspection PyShadowingNames
def line(x0, y0, x1, y1):
    steep = False
    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        steep = True
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    dx = int(x1 - x0)
    dy = int(y1 - y0)
    derror2 = abs(dy) * 2
    error2 = 0
    y = y0
    for x in range(x0, x1 + 1):
        if steep:
            yield y, x
        else:
            yield x, y
        error2 += derror2
        if error2 > dx:
            y += 1 if y1 > y0 else -1
            error2 -= dx * 2


# noinspection PyShadowingNames
def circle(x1, y1, r):
    x = 0
    y = r
    delta = 1 - 2 * r
    while y >= 0:
        yield x1 + x, y1 + y,
        yield x1 + x, y1 - y,
        yield x1 - x, y1 + y,
        yield x1 - x, y1 - y,
        error = 2 * (delta + y) - 1
        if delta < 0 and error <= 0:
            x += 1
            delta += 2 * x + 1
            continue
        error = 2 * (delta - x) - 1
        if delta > 0 and error > 0:
            y -= 1
            delta += 1 - 2 * y
            continue
        x += 1
        delta += 2 * (x - y)
        y -= 1


def triangle(t0, t1, t2):
    if t0[1] > t1[1]:
        t0, t1 = t1, t0
    if t0[1] > t2[1]:
        t0, t2 = t2, t0
    if t1[1] > t2[1]:
        t1, t2 = t2, t1

    (x0, y0), (x1, y1), (x2, y2) = t0, t1, t2
    total_height = y2 - y0
    for y in range(y0, y1 + 1):
        segment_height = y1 - y0 + 1
        alpha = float(y - y0) / total_height
        beta = float(y - y0) / segment_height
        ax = int(x0 + (x2 - x0) * alpha)
        bx = int(x0 + (x1 - x0) * beta)
        if ax > bx:
            ax, bx = bx, ax
        for j in range(ax, bx + 1):
            yield j, y


def project_with_angle(v, theta):
    # Z
    # [[cos(radians(-theta)), -sin(radians(-theta)), 0],
    #          [sin(radians(-theta)), cos(radians(-theta)), 0],
    #          [0, 0, 1]]

    arr = np.array(
        [[cos(radians(-theta)), 0, sin(radians(-theta))],
         [0, 1, 0],
         [-sin(radians(-theta)), 0, cos(radians(-theta))]],
    ).dot(
        np.transpose(np.array(v) - np.array([500, 0, 0]))
    )

    return int(arr[0]), int(arr[1])


with open('./african_head.obj') as file:
    width = 500
    height = 800

    min_x, max_x = 200, -200
    min_y, max_y = 200, -200
    min_z, max_z = 200, -200
    vertices = []
    faces = []


    def scale_pixel(x, y, z):
        scale = (min(width, height) - 1) / min((max_x - min_x), (max_y - min_y), (max_z - min_z))

        canvas_x = int((x - min_x) * scale)
        canvas_y = int((y - min_y) * scale)
        canvas_z = int((z - min_z) * scale)

        return canvas_x, height - 1 - canvas_y, canvas_z


    for line_i, obj_line in enumerate(file):
        if obj_line.startswith('v '):
            x, y, z = [float(n) for n in obj_line.split(' ')[1:]]
            vertices.append((x, y, z))

            min_x, max_x = min(min_x, x), max(max_x, x)
            min_y, max_y = min(min_y, y), max(max_y, y)
            min_z, max_z = min(min_z, z), max(max_z, z)
        elif obj_line.startswith('f '):
            f1, f2, f3 = obj_line.split(' ')[1:]
            # vertices indices start from 1 in .obj
            vi1, vi2, vi3 = [int(f.split('/')[0]) - 1 for f in [f1, f2, f3]]
            v1, v2, v3 = [scale_pixel(*vertices[vi]) for vi in [vi1, vi2, vi3]]

            faces.append((v1, v2, v3))

    while True:
        for theta in range(0, 360, 10):
            model_canvas = np.zeros((height, width, 3), dtype=np.uint8)
            for face in faces:
                v1, v2, v3 = face
                x1, y1 = project_with_angle(v1, theta)
                x2, y2 = project_with_angle(v2, theta)
                x3, y3 = project_with_angle(v3, theta)

                # cv2.line(model_canvas, (x1, y1), (x2, y2), color=(163, 142, 86))
                # cv2.line(model_canvas, (x1, y1), (x3, y3), color=(163, 142, 86))
                # cv2.line(model_canvas, (x2, y2), (x3, y3), color=(163, 142, 86))
                try:
                    for pixel in line(x1, y1, x2, y2):
                        model_canvas[pixel[::-1]] = (163, 142, 86)
                    for pixel in line(x1, y1, x3, y3):
                        model_canvas[pixel[::-1]] = (163, 142, 86)
                    for pixel in line(x2, y2, x3, y3):
                        model_canvas[pixel[::-1]] = (163, 142, 86)
                except:
                    pass
            cv2.imshow('model', model_canvas)
            cv2.waitKey(1)
