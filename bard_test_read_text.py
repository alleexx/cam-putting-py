import cv2
import numpy as np
import pytesseract

def detect_and_inspect_golf_ball(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to isolate the golf ball from the background
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Find contours (i.e., outlines) of objects in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour, which is likely to be the golf ball
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            largest_contour = contour

    # Extract the bounding rectangle of the golf ball
    (x, y, w, h) = cv2.boundingRect(largest_contour)

    # Add Scaling factor outside of detected ball area
    scaling = 20

    # Crop the golf ball from the original image
    golf_ball_image = image.copy()[y - scaling:y + h + scaling, x - scaling:x + w + scaling]

    # Show the golf ball image
    cv2.imshow('Detail View',golf_ball_image)
    # wait 10 seconds for user to press any key
    cv2.waitKey(10000)
    
    # Detect printed text on the golf ball using Tesseract OCR
    # Install Tesseract OCR beforehand: https://tesseract-ocr.github.io/
    text = pytesseract.image_to_string(golf_ball_image, config='--psm 10')
    print(f"Detected text on the golf ball: {text}")

    # Inspect the orientation of the printed text
    # Assuming the text is centered on the golf ball
    text_center_x = w // 2
    text_center_y = h // 2

    # Calculate the angle of the text based on its center coordinates
    angle = np.arctan2(text_center_y - y, text_center_x - x)
    # Convert the angle to degrees
    # angle_in_degrees = angle * 180 / np.pi

    # Mark the angle of the text
    angle_line_length = 20
    angle_line_left = int(round(x + w / 2 + angle_line_length * np.cos(angle)))
    angle_line_top = int(round(y + h / 2 + angle_line_length * np.sin(angle)))
    angle_line_right = int(round(x + w / 2))
    angle_line_bottom = int(round(y + h / 2))
    cv2.line(image, (angle_line_right,angle_line_bottom), (angle_line_left, angle_line_top), (0, 0, 255), 2)

    # Mark the horizontal line
    horizontal_line_y = int(round(y + h / 2))
    # horizontal_line_top = horizontal_line_y - 10
    # horizontal_line_bottom = horizontal_line_y + 10
    cv2.line(image, (0, horizontal_line_y), (image.shape[1], horizontal_line_y), (0, 255, 0), 2)

    # Draw a rectangle around the golf ball in the original image
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image

# Load the image
image = cv2.imread('ball-tracking-snap3.png')

# Detect and inspect the golf ball in the image
image = detect_and_inspect_golf_ball(image)

# Display the image with the golf ball highlighted
cv2.imshow('Golf Ball Detection and Printing Inspection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()