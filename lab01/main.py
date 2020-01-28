import cv2

from vector_scope import convert_to_vectorscope

if __name__ == '__main__':
    inputImage = cv2.imread('images/input.jpg')
    ycbcr, outputImage = convert_to_vectorscope(inputImage)
    ycbcr2, outputImage2 = convert_to_vectorscope(ycbcr)

    cv2.imshow('Input', inputImage)
    cv2.imshow('ycbcr', ycbcr)
    cv2.imwrite('images/output.png', outputImage)
    cv2.imshow('outputImage', outputImage2)

    cv2.waitKey(0)
