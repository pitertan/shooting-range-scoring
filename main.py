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
captured_frame = None  # Frame terakhir yang di-capture

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

def preview_and_process():
    """Show real-time preview and process captured frame in the same window."""
    global captured_frame, CENTER_X, CENTER_Y

    cap = cv2.VideoCapture(0)
    process_mode = False  # Flag to toggle between preview and processing

    while True:
        if not process_mode:
            # Read frame only if in preview mode
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
                break
            captured_frame = frame.copy()  # Save the current frame for processing

        # Get frame dimensions
        height, width, _ = captured_frame.shape
        CENTER_X, CENTER_Y = width // 2, height // 2

        if process_mode:
            # Process the captured frame
            gray_frame = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2GRAY)

            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray_frame = clahe.apply(gray_frame)

            # Apply median blur to reduce noise
            gray_frame = cv2.medianBlur(gray_frame, 5)

            # Edge detection with adjusted thresholds
            blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
            edges = cv2.Canny(blurred_frame, 50, 150)

            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            result_frame = captured_frame.copy()
            scores = []

            # Draw target center and scoring radii on the frame
            cv2.circle(result_frame, (CENTER_X, CENTER_Y), 5, (0, 255, 0), -1)
            cv2.putText(result_frame, "Target", (CENTER_X + 10, CENTER_Y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            max_radius = min(width, height) // 2
            for radius in RADIUS_SCORES:
                if radius > max_radius:
                    radius = max_radius
                cv2.circle(result_frame, (CENTER_X, CENTER_Y), radius, (255, 0, 0), 1)

            # Process each contour
            for contour in contours:
                # Filter based on contour area
                contour_area = cv2.contourArea(contour)
                if contour_area < 20 or contour_area > 1000: #(default 20)
                    continue

                # Filter based on aspect ratio
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                if aspect_ratio < 0.5 or aspect_ratio > 1.5: #(default 0,7)
                    continue

                # Filter based on circularity
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * math.pi * contour_area / (perimeter ** 2)
                if circularity < 0.2: #(default 0,4)
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

            # Show the processed frame
            cv2.imshow("Shooting Range", result_frame)

        else:
            # Preview mode
            preview_frame = captured_frame.copy()

            # Draw concentric circles for scoring zones
            max_radius = min(width, height) // 2
            for radius in RADIUS_SCORES:
                if radius > max_radius:
                    radius = max_radius
                cv2.circle(preview_frame, (CENTER_X, CENTER_Y), radius, (255, 0, 0), 1)

            # Draw target center
            cv2.circle(preview_frame, (CENTER_X, CENTER_Y), 5, (0, 255, 0), -1)
            cv2.putText(preview_frame, "Press 'C' to capture", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(preview_frame, "Press 'Q' to Quit", (470, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(preview_frame, "Press 'R' to Recapture", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Show the preview frame
            cv2.imshow("Shooting Range", preview_frame)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            print("Frame captured. Processing...")
            process_mode = True  # Switch to processing mode
        elif key == ord('r'):
            print("Returning to preview mode.")
            process_mode = False  # Return to preview mode
        elif key == ord('q'):
            print("Exiting...")
            break

    cap.release()
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

    # Capture Frame Button
    capture_button = Button(root, text="Start Detection", command=preview_and_process, width=20, height=2)
    capture_button.pack(pady=10)

    # Exit Button
    exit_button = Button(root, text="Exit", command=lambda: root.destroy(), width=20, height=2)
    exit_button.pack(pady=10)

    # Start GUI
    root.mainloop()

if __name__ == "__main__":
    main_gui()
