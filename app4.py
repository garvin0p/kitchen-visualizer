# import cv2
# import numpy as np
# from matplotlib import pyplot as plt

# def process_image(image):
#     # Convert to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Apply GaussianBlur to reduce noise and improve edge detection
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
#     # Apply Canny Edge Detection
#     edges = cv2.Canny(blur, 50, 150)
    
#     # Dilate the edges to close gaps
#     dilated = cv2.dilate(edges, None, iterations=2)
#     return dilated

# def detect_countertops(image, processed_image):
#     # Find contours
#     contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     countertop_contours = []
#     for contour in contours:
#         epsilon = 0.02 * cv2.arcLength(contour, True)
#         approx = cv2.approxPolyDP(contour, epsilon, True)
#         if len(approx) == 4:  # Looking for rectangles
#             area = cv2.contourArea(contour)
#             if area > 10000:  # Filter small areas
#                 countertop_contours.append(contour)
#     return countertop_contours

# def apply_texture(image, contours, texture_color=(0, 0, 0)):
#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         image[y:y+h, x:x+w] = texture_color
#     return image

# # Load the image
# image_path = "kitchen1.jpg"
# image = cv2.imread(image_path)

# # Process the image to find edges
# processed_image = process_image(image)

# # Detect countertops based on the processed image
# countertop_contours = detect_countertops(image, processed_image)

# # Apply a black texture to the detected countertops
# texture_color = (0, 0, 0)  # Black color for the texture
# result_image = apply_texture(image.copy(), countertop_contours, texture_color)

# # Display the result
# plt.figure(figsize=(10, 10))
# plt.subplot(1, 2, 1)
# plt.title("Original Image")
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.subplot(1, 2, 2)
# plt.title("Texture Applied")
# plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
# plt.show()


import cv2
import numpy as np
from matplotlib import pyplot as plt

def process_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise and improve edge detection
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny Edge Detection
    edges = cv2.Canny(blur, 50, 150)
    
    # Dilate the edges to close gaps
    dilated = cv2.dilate(edges, None, iterations=2)
    return dilated

def detect_countertops(image, processed_image):
    # Find contours
    contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    countertop_contours = []
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:  # Looking for rectangles
            area = cv2.contourArea(contour)
            if area > 10000:  # Filter small areas
                countertop_contours.append(contour)
    return countertop_contours

def apply_texture(image, contours, texture):
    texture = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)
    for contour in contours:
        mask = np.zeros_like(image)
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
        x, y, w, h = cv2.boundingRect(contour)
        resized_texture = cv2.resize(texture, (w, h))
        
        # Only replace the pixels in the region of interest (the countertop area)
        roi = image[y:y+h, x:x+w]
        mask_roi = mask[y:y+h, x:x+w]
        texture_area = cv2.bitwise_and(resized_texture, mask_roi)
        image[y:y+h, x:x+w] = cv2.addWeighted(roi, 0, texture_area, 1, 0)
    return image

# Load the original image
image_path = "kitchen1.jpg"
image = cv2.imread(image_path)

# Load the texture image
texture_path = "static/1.webp"  # Make sure this is a marble texture image
texture = cv2.imread(texture_path)

# Process the image to find edges
processed_image = process_image(image)

# Detect countertops based on the processed image
countertop_contours = detect_countertops(image, processed_image)

# Apply the texture to the detected countertops
result_image = apply_texture(image.copy(), countertop_contours, texture)

# Display the result
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.subplot(1, 2, 2)
plt.title("Texture Applied")
plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
plt.show()
