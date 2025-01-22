import cv2
import numpy as np
import math
import tkinter as tk
from tkinter import Tk, Label, Button, filedialog

# Global variables
CENTER_X, CENTER_Y = 0, 0
RADIUS_SCORES = []
SCORES = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
template_image = None
selected_image = None  # Gambar yang dipilih untuk diproses

def load_template():
    global CENTER_X, CENTER_Y, RADIUS_SCORES, template_image
    template_image = filedialog.askopenfilename(
        title="Select Template",
        filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")]
    )
    if not template_image:
        print("No template selected.")
        return
    template_image = cv2.imread(template_image)
    height, width, _ = template_image.shape
    CENTER_X, CENTER_Y = width // 2, height // 2
    RADIUS_SCORES = [i * (width // 20) for i in range(1, 11)]
    print(f"Template loaded: {template_image}")
    print(f"Center: ({CENTER_X}, {CENTER_Y})")
    print(f"Radius Scores: {RADIUS_SCORES}")

def load_image():
    """Load image manually and process it."""
    global captured_frame, template_image, CENTER_X, CENTER_Y

    image_path = filedialog.askopenfilename(
        title="Select Image to Process",
        filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")]
    )
    if not image_path:
        print("No image selected.")
        return

    # Load the image
    uploaded_image = cv2.imread(image_path)
    if uploaded_image is None:
        print("Failed to load image.")
        return

    # Resize uploaded image to match the template size
    if template_image is not None:
        template_height, template_width = template_image.shape[:2]
        uploaded_image = cv2.resize(uploaded_image, (template_width, template_height))
        print(f"Uploaded image resized to match template: {template_width}x{template_height}")

    captured_frame = uploaded_image  # Store for processing

    # Update center and display image
    CENTER_X, CENTER_Y = template_width // 2, template_height // 2
    process_image(captured_frame)

def process_image(frame):
    """Process the given frame to detect shots and calculate scores."""
    global CENTER_X, CENTER_Y

    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_frame = clahe.apply(gray_frame)

    # Apply median blur to reduce noise
    gray_frame = cv2.medianBlur(gray_frame, 5)

    # Edge detection
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    edges = cv2.Canny(blurred_frame, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result_frame = frame.copy()
    scores = []

    # Draw target center and scoring radii on the frame
    cv2.circle(result_frame, (CENTER_X, CENTER_Y), 5, (0, 255, 0), -1)
    cv2.putText(result_frame, "Target", (CENTER_X + 10, CENTER_Y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    max_radius = min(result_frame.shape[1], result_frame.shape[0]) // 2
    for radius in RADIUS_SCORES:
        if radius > max_radius:
            radius = max_radius
        cv2.circle(result_frame, (CENTER_X, CENTER_Y), radius, (255, 0, 0), 1)

    # Process each contour
    for contour in contours:
        # Filter based on contour area
        contour_area = cv2.contourArea(contour)
        if contour_area < 20 or contour_area > 1000:
            continue

        # Filter based on aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        if aspect_ratio < 0.5 or aspect_ratio > 1.5:
            continue

        # Filter based on circularity
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * math.pi * contour_area / (perimeter ** 2)
        if circularity < 0.2:
            continue

        # Calculate score
        (x, y), radius = cv2.minEnclosingCircle(contour)
        x, y = int(x), int(y)
        score = calculate_score(x, y)
        scores.append(score)

        # Draw detected dot and score
        cv2.circle(result_frame, (x, y), 5, (0, 0, 255), -1)
        cv2.line(result_frame, (CENTER_X, CENTER_Y), (x, y), (0, 255, 255), 2)
        cv2.putText(result_frame, f"{score}", (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    total_score = sum(scores)
    cv2.putText(result_frame, f"Total Score: {total_score}", (10, result_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the result
    cv2.imshow("Processed Image", result_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def calculate_score(x, y):
    distance = calculate_distance(x, y, CENTER_X, CENTER_Y)
    for i, radius in enumerate(RADIUS_SCORES):
        if distance <= radius:
            return SCORES[i]
    return 0

def main_gui():
    root = tk.Tk()
    root.title("Shooting Range Scoring System")

    # Label for the title
    label = Label(root, text="Shooting Range Scoring System", font=("Helvetica", 16))
    label.pack(pady=10)

    # Load Template Button
    load_template_button = Button(root, text="Load Template", command=load_template, width=20, height=2)
    load_template_button.pack(pady=10)

    # Upload Image Button
    upload_button = Button(root, text="Upload Image", command=load_image, width=20, height=2)
    upload_button.pack(pady=10)

    # Exit Button
    exit_button = Button(root, text="Exit", command=lambda: root.destroy(), width=20, height=2)
    exit_button.pack(pady=10)

    # Start GUI
    root.mainloop()


if __name__ == "__main__":
    main_gui()
