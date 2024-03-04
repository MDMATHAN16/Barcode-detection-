import numpy as np
import argparse
from google.colab.patches import cv2_imshow
import imutils
import cv2

# Upload your image to Google Colab and provide the correct path
image_path = '/content/WhatsApp Image 2023-08-09 at 10.39.46 PM (1).jpeg'

# Load the image
image = cv2.imread(image_path)

# Convert the image to grayscale for easier processing
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply edge detection to find the boundaries of the box
edges = cv2.Canny(gray, 125,125)  # Define threshold values

# Find contours in the edges image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterate through the contours to find the largest box (assumed as the top box)
largest_contour = max(contours, key=cv2.contourArea)

# Find the center of the largest box
M = cv2.moments(largest_contour)
center_x = int(M['m10'] / M['m00'])
center_y = int(M['m01'] / M['m00'])

# Draw the box's border and mark its center
cv2.drawContours(image, [largest_contour], -1, (0, 255, 0), 2)
cv2.circle(image, (center_x, center_y), 5, (0, 0, 255), -1)

# Get the bounding rectangle of the largest box
x, y, w, h = cv2.boundingRect(largest_contour)

# Extract the region of interest (ROI) within the box
box_roi = image[y:y+h, x:x+w]
# barcode detection
img=box_roi
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
gradX = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=-1)
gradY = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=-1)
gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)

blurred = cv2.blur(gradient, (9, 9))
(_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)


kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

closed = cv2.erode(closed, None, iterations = 4)
closed = cv2.dilate(closed, None, iterations = 4)

cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]

rect = cv2.minAreaRect(c)
area = cv2.contourArea(c)
box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
box = np.int0(box)
if area>1000:
  print("Barcode Detected")
  cv2.drawContours(img, [box], -1, (255, 0, 0), 3)

# Display the modified image with Barcode
cv2_imshow(image)
