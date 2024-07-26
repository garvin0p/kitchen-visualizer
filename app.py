from flask import Flask, request, jsonify, render_template, url_for
import cv2
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

# Ensure the static/uploads directory exists
os.makedirs('static/uploads', exist_ok=True)

# Function to apply a texture to a specified region
def apply_texture(image, texture_path, coords):
    texture = Image.open(texture_path)
    x1, y1, x2, y2 = coords
    texture_resized = texture.resize((x2 - x1, y2 - y1))
    texture_array = np.array(texture_resized)

    if len(texture_array.shape) == 2:  # If texture is grayscale, convert to 3 channels
        texture_array = cv2.cvtColor(texture_array, cv2.COLOR_GRAY2BGR)
    
    # Ensure both image and texture have the same number of channels
    if image.shape[2] == 3 and texture_array.shape[2] == 4:  # RGBA to RGB
        texture_array = cv2.cvtColor(texture_array, cv2.COLOR_RGBA2RGB)
    elif image.shape[2] == 4 and texture_array.shape[2] == 3:  # RGB to RGBA
        texture_array = cv2.cvtColor(texture_array, cv2.COLOR_RGB2RGBA)

    # Resize the texture to match the image region
    texture_array = cv2.resize(texture_array, (x2 - x1, y2 - y1))
    image_region = image[y1:y2, x1:x2]

    if image_region.shape[:2] != texture_array.shape[:2]:
        texture_array = cv2.resize(texture_array, (image_region.shape[1], image_region.shape[0]))

    if image_region.shape[2] != texture_array.shape[2]:
        raise ValueError("The number of channels in the image region and texture array do not match")

    blended = cv2.addWeighted(image_region, 0.5, texture_array, 0.5, 0)
    image[y1:y2, x1:x2] = blended
    
    return image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Convert to grayscale and apply edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    countertop_coords = []

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) >= 4:  # Looking for quadrilateral shapes
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 1.5 < aspect_ratio < 5.0 and w > 50 and h > 50:  # Filter based on aspect ratio and size
                countertop_coords.append((x, y, x + w, y + h))

    result_path = 'static/uploads/uploaded_image.jpg'
    cv2.imwrite(result_path, img)

    return jsonify({"coords": countertop_coords, "image_url": url_for('static', filename='uploads/uploaded_image.jpg')})

@app.route('/apply_texture', methods=['POST'])
def apply_texture_to_image():
    file = request.files['image']
    texture_name = request.form['texture']
    coords = request.form.getlist('coords', type=int)

    # Convert coordinates to tuples
    coords = [(coords[i], coords[i+1], coords[i+2], coords[i+3]) for i in range(0, len(coords), 4)]

    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Load the selected texture
    texture_path = f'static/{texture_name}.webp'
    
    # Apply the texture to the detected countertops
    for coord in coords:
        img = apply_texture(img, texture_path, coord)

    # Define the result path
    result_path = 'static/uploads/result.jpg'

    # Remove existing result image if it exists
    if os.path.exists(result_path):
        os.remove(result_path)

    # Save the resulting image
    cv2.imwrite(result_path, img)

    return jsonify({"url": url_for('static', filename='uploads/result.jpg')})

if __name__ == '__main__':
    app.run(debug=True)
