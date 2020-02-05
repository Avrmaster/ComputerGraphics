import cv2
import numpy as np
from math import cos, sin, radians
from random import randint


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


def project_with_angle(v, thetaY, thetaZ, origin=(0, 0, 0)):
    arr = np.array(
        # Y
        [[cos(radians(-thetaY)), 0, sin(radians(-thetaY))],
         [0, 1, 0],
         [-sin(radians(-thetaY)), 0, cos(radians(-thetaY))]],
    ).dot(
        # Z
        np.array([
            [cos(radians(-thetaZ)), -sin(radians(-thetaZ)), 0],
            [sin(radians(-thetaZ)), cos(radians(-thetaZ)), 0],
            [0, 0, 1]
        ])
    ).dot(
        np.transpose(np.array(v) - np.array(origin))
    ) + np.array(origin)

    return arr[0], arr[1]


with open('./african_head.obj') as file:
    width = 500
    height = 800

    min_x, max_x = 200, -200
    min_y, max_y = 200, -200
    min_z, max_z = 200, -200
    vertices = []
    faces = []


    def scale_pixel(x, y):
        scale = (min(width, height) - 1) / min((max_x - min_x), (max_y - min_y))

        canvas_x = int((x - min_x) * scale)
        canvas_y = int((y - min_y) * scale)

        return canvas_x, height - 1 - canvas_y


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
            face_color = (randint(0, 255), randint(0, 255), randint(0, 255))
            faces.append((tuple(vertices[vi] for vi in [vi1, vi2, vi3]), face_color))

    thetaZ = 0
    frames = []
    for thetaX in range(0, 360, 2):
        print(f'\r{thetaX}/{360}', end='')
        thetaZ += 4

        model_canvas = np.zeros((height, width, 3), dtype=np.uint8)

        for face in faces:
            (v1, v2, v3), face_color = face

            origin = [-1, 2, 0]

            x1, y1 = scale_pixel(*project_with_angle(v1, thetaX, thetaZ, origin))
            x2, y2 = scale_pixel(*project_with_angle(v2, thetaX, thetaZ, origin))
            x3, y3 = scale_pixel(*project_with_angle(v3, thetaX, thetaZ, origin))

            # cv2.line(model_canvas, (x1, y1), (x2, y2), color=(163, 142, 86))
            # cv2.line(model_canvas, (x1, y1), (x3, y3), color=(163, 142, 86))
            # cv2.line(model_canvas, (x2, y2), (x3, y3), color=(163, 142, 86))

            cv2.drawContours(
                model_canvas,
                [np.array([[x1, y1], [x2, y2], [x3, y3]])],
                0, face_color, -1
            )

        frames.append(model_canvas)
    while True:
        for frame in frames:
            cv2.imshow('model', frame)
            cv2.waitKey(16)
