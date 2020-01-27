import cv2

from vector_scope import convert_to_vectorscope

if __name__ == '__main__':
    inputImage = cv2.imread('images/input.png')
    outputImage = convert_to_vectorscope(inputImage)
    cv2.imwrite('images/output.png', outputImage)
    cv2.imshow('Input', inputImage)
    cv2.imshow('Output', outputImage)
    cv2.waitKey(0)
