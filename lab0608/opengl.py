import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import gluPerspective

from object3D import torus


def draw_torus(R, r, vertices_count=20):
    glBegin(GL_TRIANGLE_STRIP)
    for face in torus(R, r, vertices_count):
        v1, v2, v3 = face
        glVertex3fv(v1)
        glVertex3fv(v2)
        glVertex3fv(v3)
    glEnd()


def main():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    gluPerspective(45, (display[0] / display[1]), 0.1, 500.0)
    glTranslatef(0.0, 0.0, -200)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        glRotatef(2, 3, 1, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        draw_torus(50, 20)
        pygame.display.flip()
        pygame.time.wait(8)


main()
