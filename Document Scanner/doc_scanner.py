import cv2
import numpy as np
from skimage.filters import threshold_local
import imutils
import streamlit as st

# Helper function for four-point transformation
def four_point_transform(image, pts):
    rect = np.array(pts, dtype="float32")
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

# Streamlit app title


# Load the image
image = cv2.imread("60c4199364474569561cba359d486e6c69ae8cba.jpeg")
orig = image.copy()

# Resize the image
image = imutils.resize(image, height=500)
ratio = orig.shape[0] / 500.0

# Convert to grayscale and detect edges
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

# Find contours and select the largest one that looks like a document
contours = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
countour_image = image.copy()

for contour in contours:
    # Approximate the contour to a polygon
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    
    # If the approximated contour has four points, we have found our document
    if len(approx) == 4:
        document_contour = approx
        break

document_contour = document_contour.reshape(4, 2) * ratio
warped = four_point_transform(orig, document_contour)
warped = cv2.resize(warped,(557,480))
warped = cv2.rotate(warped,cv2.ROTATE_90_COUNTERCLOCKWISE)

cv2.drawContours(countour_image,contours,-1,(0,255.0,0),2)

cv2.imshow('image',image)
cv2.imshow('image',edged)
cv2.imshow('image',countour_image)
cv2.imshow('image',cv2.flip(warped,0))

cv2.waitKey(0)
cv2.destroyAllWindows()