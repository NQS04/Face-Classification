import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read Image
img = cv2.imread('img1.jpg') 
if img is None:
    print("Can not file img1.jpg")
    exit()

# show img information
print("Image size:", img.shape)
print("Data type:", img.dtype)

# Convert to RGB to show actual color in matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Show original image and grayscale image
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(img_rgb)
plt.title("Original")
plt.subplot(1,2,2)
plt.imshow(gray, cmap='gray')
plt.title("Grayscale")
plt.show()

# Resize and blur photo
resized = cv2.resize(img, (200, 200))
blurred = cv2.GaussianBlur(img, (11, 11), 0)
output1 = cv2.imwrite("output_resized.jpg", resized)
output2 = cv2.imwrite("output_blurred.jpg", blurred)

if output1 and output2:
    print("Saved resized and blurred.")
