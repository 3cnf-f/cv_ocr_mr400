import pytesseract as tess
import cv2
import numpy as np

image = cv2.imread("output.png")
text = tess.image_to_string(image)
print(text)
