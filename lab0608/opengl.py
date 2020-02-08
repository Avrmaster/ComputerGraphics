import pygame
from pygame.locals import *
from OpenGL.GL import *

from object3D import torus, rotate
import numpy as np


def draw_torus(R, r, vertices_count=15):
    glBegin(GL_TRIANGLES)
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
        glNormal3f(*normal_vector.tolist())

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

    light_angle = [0, 0, -1]
    while True:
        light_angle = rotate(light_angle, 0.1, 0.05, 0.025)

        glLightfv(GL_LIGHT0, GL_POSITION, light_angle + [0])

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        glRotatef(2, 1, 1, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        draw_torus(300, 60)
        pygame.display.flip()
        pygame.time.wait(16)


main()
