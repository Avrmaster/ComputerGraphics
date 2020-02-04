import cv2
import numpy as np
from random import randint


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


with open('./african_head.obj') as file:
    width = 600
    height = 800
    model_canvas = np.zeros((height, width, 3), dtype=np.uint8)
    model_canvas_triangles = np.zeros((height, width, 3), dtype=np.uint8)
    model_canvas_cv2 = np.zeros((height, width, 3), dtype=np.uint8)
    model_canvas_triangles_cv2 = np.zeros((height, width, 3), dtype=np.uint8)
    min_x = 200
    max_x = -200
    min_y = 200
    max_y = -200
    vertices = []


    def scale_pixel(x, y):
        x_std = (x - min_x) / (max_x - min_x)
        y_std = (y - min_y) / (max_y - min_y)
        canvas_x = int(x_std * width)
        canvas_y = int(y_std * height)
        return min(canvas_x, width - 1), min(height - canvas_y, height - 1)


    for line_i, obj_line in enumerate(file):
        if obj_line.startswith('v '):
            x, y = [float(n) for n in obj_line.split(' ')[1:3]]
            vertices.append((x, y))
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
        elif obj_line.startswith('f '):
            f1, f2, f3 = obj_line.split(' ')[1:]
            v1, v2, v3 = [int(f.split('/')[0]) for f in [f1, f2, f3]]

            x1, y1 = scale_pixel(*vertices[v1 - 1])
            x2, y2 = scale_pixel(*vertices[v2 - 1])
            x3, y3 = scale_pixel(*vertices[v3 - 1])

            for pixel in line(x1, y1, x2, y2):
                model_canvas[pixel[::-1]] = (163, 142, 86)
            for pixel in line(x1, y1, x3, y3):
                model_canvas[pixel[::-1]] = (229, 220, 146)
            for pixel in line(x2, y2, x3, y3):
                model_canvas[pixel[::-1]] = (229, 222, 82)

            cv2.line(model_canvas_cv2, (x1, y1), (x2, y2), (247, 255, 252))
            cv2.line(model_canvas_cv2, (x1, y1), (x3, y3), (160, 160, 33))
            cv2.line(model_canvas_cv2, (x2, y2), (x3, y3), (101, 104, 4))

            triangle_color = (randint(0, 255), randint(0, 255), randint(0, 255))
            for pixel in triangle((x1, y1), (x2, y2), (x3, y3)):
                model_canvas_triangles[pixel[::-1]] = triangle_color
            cv2.drawContours(
                model_canvas_triangles_cv2,
                [np.array([[x1, y1], [x2, y2], [x3, y3]])],
                0, triangle_color, -1
            )

            if line_i % 15 == 0:
                # pass
                cv2.imshow('model_canvas_cv2', model_canvas_cv2)
                cv2.imshow('model_triangles', model_canvas_triangles)
                cv2.imshow('model_triangles_cv2', model_canvas_triangles_cv2)
                cv2.imshow('model', model_canvas)
                cv2.waitKey(1)


def plot(cnv, x, y):
    cnv[y, x] = (200, 201, 20)


test_canvas = np.zeros((400, 400, 3), dtype=np.uint8)
for pixel in line(0, 200, 100, 120):
    plot(test_canvas, *pixel)

for pixel in circle(100, 250, 20):
    plot(test_canvas, *pixel)

for pixel in triangle((0, 0), (399, 399), (250, 60)):
    plot(test_canvas, *pixel)

cv2.imshow('test_canvas', test_canvas)
cv2.waitKey(0)
