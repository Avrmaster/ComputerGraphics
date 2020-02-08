import pygame
from pygame.locals import *
from OpenGL.GL import *

from object3D import torus, rotate
import numpy as np


def gen_torus(R, r, vertices_count=10):
    faces_normals = []
    for face in torus(R, r, vertices_count):
        v1, v2, v3 = face
        x1, y1, z1 = v1
        x2, y2, z2 = v2
        x3, y3, z3 = v3

        normal_vector: np.ndarray = np.array([
            (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1),
            (z2 - z1) * (x3 - x1) - (x2 - x1) * (z3 - z1),
            (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
        ])
        faces_normals.append(((v1, v2, v3), normal_vector.tolist()))
    return faces_normals


def draw_torus(generated_torus):
    glBegin(GL_TRIANGLES)
    for face, normal in generated_torus:
        v1, v2, v3 = face
        glNormal3f(*normal)

        glColor3fv((0.3, 0.8, 0.5))
        glVertex3fv(v1)

        glColor3fv((0.9, 0.3, 0.5))
        glVertex3fv(v2)

        glColor3fv((0.3, 0.5, 0.9))
        glVertex3fv(v3)
    glEnd()


def main():
    width = 1000
    height = 800

    # Viewport Init
    pygame.init()
    pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
    glOrtho(0, width, 0, height, 2000, -2000)
    glTranslatef(width / 2, height / 2, 0)

    #

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_BLEND)
    glEnable(GL_NORMALIZE)
    glShadeModel(GL_SMOOTH)

    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    # glColorMaterial(GL_FRONT_AND_BACK, GL_SPECULAR)
    # glMateriali(GL_FRONT_AND_BACK, GL_SHININESS, 120)

    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1])
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [2, 2, 2, 1])
    # glLightfv(GL_LIGHT0, GL_SPECULAR, [0.4, 0.4, 0.4, 0.2])

    toruses = [
        gen_torus(300, 60),
        gen_torus(180, 45),
        gen_torus(90, 30)
    ]
    rotations = [0 for t in toruses]

    light_angle = [0, 0, -1]
    while True:
        light_angle = rotate(light_angle, 0.1, 0.05, 0.025)

        glLightfv(GL_LIGHT0, GL_POSITION, light_angle + [0])

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        for tor_i, tor in enumerate(toruses):
            glPushMatrix()
            rotations[tor_i] += (tor_i + 1) * 1.5
            glRotatef(rotations[tor_i], 1, 1, 1)
            draw_torus(tor)
            glPopMatrix()

        pygame.display.flip()
        pygame.time.wait(16)


if __name__ == '__main__':
    main()
