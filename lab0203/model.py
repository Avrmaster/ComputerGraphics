import cv2
import numpy as np
from math import cos, sin, radians
from random import randint
from time import time
from playsound import playsound

light_vector = np.array((0., -1., 0.))
view_dir = np.array((0., 0., -1.))
ambient_coef = 0.1
diffuse_coef = 0.8
specular_coef = 20


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


# noinspection PyShadowingNames
def project_with_angle(v, theta_x, theta_y, theta_z, origin=(0, 0, 0)):
    arr = np.array(
        # X
        [[1, 0, 0],
         [0, cos(radians(-theta_x)), -sin(radians(-theta_x))],
         [0, sin(radians(-theta_x)), cos(radians(-theta_x))]],
    ).dot(
        # Y
        np.array([
            [cos(radians(-theta_y)), 0, sin(radians(-theta_y))],
            [0, 1, 0],
            [-sin(radians(-theta_y)), 0, cos(radians(-theta_y))]
        ])
    ).dot(
        # Z
        np.array([
            [cos(radians(-theta_z)), -sin(radians(-theta_z)), 0],
            [sin(radians(-theta_z)), cos(radians(-theta_z)), 0],
            [0, 0, 1]
        ])
    ).dot(
        np.transpose(np.array(v) - np.array(origin))
    ) + np.array(origin)

    return arr[0], arr[1], arr[2]


# noinspection PyShadowingNames
def get_z_order(face):
    (v1, v2, v3), face_color = face
    return v1[2] + v2[2] + v3[2]


def get_lighten_color(face, original_color):
    v1, v2, v3 = face
    x1, y1, z1 = v1
    x2, y2, z2 = v2
    x3, y3, z3 = v3

    # diffuse
    normal_vector: np.ndarray = np.array([
        (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1),
        (z2 - z1) * (x3 - x1) - (x2 - x1) * (z3 - z1),
        (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    ])
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    diffuse_light = max(light_vector.dot(normal_vector), 0.0)

    # specular
    reflect_dir = 2 * normal_vector * (
            normal_vector.dot(light_vector) / normal_vector.dot(normal_vector)) - light_vector
    specular_light = pow(max(view_dir.dot(reflect_dir), 0.0), 32)

    return tuple(
        min(255, int(c * (
                ambient_coef +
                diffuse_coef * diffuse_light +
                specular_coef * specular_light
        ))) for c in original_color
    )


with open('./african_head.obj') as file:
    width = 500
    height = 800

    min_x, max_x = 200, -200
    min_y, max_y = 200, -200
    min_z, max_z = 200, -200
    vertices = []
    faces = []


    def scale_pixel(x, y, z):
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

    theta_x = 0
    theta_z = 0
    frames = []
    origin = [-1, 2, 0]

    for theta_y in range(0, 360):
        print(f'\r{theta_y + 1}/{360}', end='')
        theta_z += 3.19
        theta_x += 2.2

        model_canvas = np.zeros((height, width, 3), dtype=np.uint8)
        rotated_faces = [
            (
                [project_with_angle(v, theta_x, theta_y, theta_z, origin) for v in vs],
                c
            )
            for (vs, c)
            in faces
        ]

        ordered_faces = reversed(sorted(rotated_faces, key=get_z_order))

        for face in ordered_faces:
            (v1, v2, v3), face_color = face

            x1, y1 = scale_pixel(*v1)
            x2, y2 = scale_pixel(*v2)
            x3, y3 = scale_pixel(*v3)

            cv2.line(model_canvas, (x1, y1), (x2, y2), color=(163, 142, 86))
            cv2.line(model_canvas, (x1, y1), (x3, y3), color=(163, 142, 86))
            cv2.line(model_canvas, (x2, y2), (x3, y3), color=(163, 142, 86))

            cv2.drawContours(
                model_canvas,
                [np.array([[x1, y1], [x2, y2], [x3, y3]])],
                0, get_lighten_color((v1, v2, v3), face_color), -1
            )

        frames.append(model_canvas)

    playsound('./Scatman.mp3', block=False)
    while True:
        for frame in frames + list(reversed(frames)):
            start = int(time() * 1000)
            cv2.imshow('model', frame)
            end = int(time() * 1000)
            cv2.waitKey(max(1, 16 - (start - end)))
