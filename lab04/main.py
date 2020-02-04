import cv2
import numpy as np
import colorsys

width = 600
height = 600
canvas = np.zeros((height, width, 3), dtype=np.uint8)


def dots(p, q, max_n, xn=0, yn=0, n=0):
    if n > max_n:
        return
    xn1 = xn ** 2 - yn ** 2 + p
    yn1 = 2 * xn * yn + q
    yield xn1, yn1
    for new_dot in dots(p, q, max_n, xn1, yn1, n + 1):
        yield new_dot


pmin, pmax, qmin, qmax = -3, 3, -2, 2
max_iterations = 122
infinity_border = 10

for ip, p in enumerate(np.linspace(pmin, pmax, width)):
    for iq, q in enumerate(np.linspace(qmin, qmax, height)):
        c = p + 1j * q

        z = 0
        for k in range(max_iterations):
            z = z ** 2 + c

            if abs(z) > infinity_border:
                canvas[ip, iq] = (k * 2, 150, 255)
                break

cv2.imshow('fractal', canvas)
cv2.waitKey(1)

# for (x, y) in dots(beautiful_q, beautiful_p, 200):
#     print(x, y)
# canvas_x = int(width * (x + 0.5) / 100)
# canvas_y = int(height * (y + 0.5) / 100)
# canvas[canvas_y, canvas_x] = (255, 255, 255)

output = cv2.cvtColor(canvas, cv2.COLOR_HSV2BGR)
cv2.imshow('fractal', output)
cv2.waitKey(0)
