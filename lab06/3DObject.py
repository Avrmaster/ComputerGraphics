import numpy as np
import cv2
from math import cos, sin, pi, radians
from random import randint

light_vector = np.array((0., -1., 0.))
view_dir = np.array((0., 0., -1.))
ambient_coef = 0.1
diffuse_coef = 0.8
specular_coef = 20


def move_cycle(arr: list):
    copy = arr[1:]
    return copy + [arr[0]]


def get_z_order(face):
    (v1, v2, v3), face_color = face
    return min(v1[2], v2[2], v3[2])


def rotate(vertex, theta_x=0., theta_y=0., theta_z=0.):
    arr: np.ndarray = np.array(
        # X
        [[1, 0, 0],
         [0, cos(-theta_x), -sin(-theta_x)],
         [0, sin(-theta_x), cos(-theta_x)]],
    ).dot(
        # Y
        np.array([
            [cos(-theta_y), 0, sin(-theta_y)],
            [0, 1, 0],
            [-sin(-theta_y), 0, cos(-theta_y)]
        ])
    ).dot(
        # Z
        np.array([
            [cos(-theta_z), -sin(-theta_z), 0],
            [sin(-theta_z), cos(-theta_z), 0],
            [0, 0, 1]
        ])
    ).dot(
        np.transpose(np.array(vertex))
    )
    return arr.tolist()


def translate(vertex, vector):
    return (np.array(vertex) + np.array(vector)).tolist()


def circle(r, vertices_count):
    for i in range(vertices_count):
        angle = 2 * pi * i / vertices_count
        yield int(r * cos(angle)), int(r * sin(angle)), 0


# noinspection PyPep8Naming
def torus(R, r, vertices_count):
    cut_circles = []
    for i in range(vertices_count):
        angle = 2 * pi * i / vertices_count
        translate_vector = [R * cos(angle), 0, R * sin(angle)]
        cut_circles.append(
            [
                translate(rotate(vertex, theta_y=angle), translate_vector)
                for vertex
                in circle(r, vertices_count)
            ]
        )
    faces = []
    for circle1, circle2 in zip(cut_circles, move_cycle(cut_circles)):
        for v1, v2, v3, v4 in zip(circle1, move_cycle(circle1), circle2, move_cycle(circle2)):
            faces.extend([
                (v1, v2, v3),
                (v2, v4, v3)
            ])
    return faces


# noinspection PyShadowingNames
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


canvas_width = 600
canvas_height = 600
torus_translate = [300, 300, 0]
torus_faces = []
for face in torus(R=250, r=60, vertices_count=20):
    face_color = (randint(0, 255), randint(0, 255), randint(0, 255))
    torus_faces.append((face, face_color))

angle_deg = 0
while True:
    angle_deg += 0.03

    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    angle = radians(angle_deg)

    transformed_faces = []

    for face, color in torus_faces:
        v1, v2, v3 = [rotate(v, theta_x=(pi / 3), theta_y=angle_deg, theta_z=angle_deg) for v in face]
        v1, v2, v3 = [translate(v, torus_translate) for v in [v1, v2, v3]]
        transformed_faces.append(((v1, v2, v3), color))

    ordered_faces = reversed(sorted(transformed_faces, key=get_z_order))

    for face_index, (face, original_color) in enumerate(ordered_faces):
        (x1, y1, z1), \
        (x2, y2, z2), \
        (x3, y3, z3) = face

        lighten_color = get_lighten_color(face, original_color)

        cv2.drawContours(
            canvas,
            [np.array([[int(x1), int(y1)], [int(x2), int(y2)], [int(x3), int(y3)]])],
            0, lighten_color, -1
        )

    cv2.imshow('thor', canvas)
    cv2.waitKey(1)
