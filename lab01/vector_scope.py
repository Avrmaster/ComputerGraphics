import numpy as np
import cv2


def convert_to_vectorscope(im: np.ndarray):
    vector_image: np.ndarray = cv2.imread('images/vectorscopeBackground.png')
    ycbcr = np.zeros(im.shape, dtype=im.dtype)

    for row_ind, row in enumerate(im):
        print(f'\rProcessing: {int(100 * row_ind / (im.shape[0] - 1))}%', end='')
        for col_ind, pixel in enumerate(row):
            fb, fg, fr = pixel
            y = int(0.299 * fr + 0.587 * fg + 0.114 * fb)  # 0-255
            cb = -0.169 * fr - 0.331 * fg + 0.500 * fb  # -127 - 127
            cr = 0.500 * fr - 0.418 * fg - 0.082 * fb  # -127 - 127

            ycbcr[row_ind, col_ind] = [y, int(cb + 128), int(cr + 128)]

            # cords on the plot
            vx = int(vector_image.shape[0] * (0.5 + cb / 255))
            vy = int(vector_image.shape[1] * (0.5 - cr / 255))

            cv2.circle(vector_image, (vx, vy), 1, color=(0, round(y), 0))
    print()

    return ycbcr, vector_image
