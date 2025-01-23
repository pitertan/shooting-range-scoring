import cv2
import numpy as np
import math
import tkinter as tk
from tkinter import Tk, Label, Button, filedialog

# Global variables
CENTER_X, CENTER_Y = 0, 0
RADIUS_SCORES = []
SCORES = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
INITIAL_SHOTS = []  # List untuk menyimpan koordinat tembakan awal
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

def filter_contour(contour):
    """Apply filtering criteria to a contour and return True if it's valid."""
    contour_area = cv2.contourArea(contour)
    if contour_area < 20 or contour_area > 1000:  # Filter based on area
        return False

    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    if aspect_ratio < 0.5 or aspect_ratio > 1.5:  # Filter based on aspect ratio
        return False

    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * math.pi * contour_area / (perimeter ** 2)
    if circularity < 0.2:  # Filter based on circularity
        return False

    return True


def detect_initial_shots():
    """Detect initial shots on the target and store their coordinates."""
    global INITIAL_SHOTS, captured_frame, CENTER_X, CENTER_Y

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Detecting initial shots. Press 'C' to capture and save initial state. Press 'Q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break
        captured_frame = frame.copy()

        # Get frame dimensions
        height, width, _ = captured_frame.shape
        CENTER_X, CENTER_Y = width // 2, height // 2

        # Draw concentric circles for scoring zones (same as preview_and_process)
        max_radius = min(width, height) // 2
        for radius in RADIUS_SCORES:
            if radius > max_radius:
                radius = max_radius
            cv2.circle(captured_frame, (CENTER_X, CENTER_Y), radius, (255, 0, 0), 1)

        # Draw target center (same as preview_and_process)
        cv2.circle(captured_frame, (CENTER_X, CENTER_Y), 5, (0, 255, 0), -1)
        cv2.putText(captured_frame, "Target", (CENTER_X + 10, CENTER_Y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display the current frame
        cv2.imshow("Initial Shots Detection", captured_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):  # Capture and process
            print("Capturing initial state...")
            gray_frame = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.medianBlur(gray_frame, 5)
            edges = cv2.Canny(gray_frame, 50, 150)

            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Detect and store initial shot coordinates
            INITIAL_SHOTS.clear()
            for contour in contours:
                if not filter_contour(contour):  # Use the same filter function
                    continue

                (x, y), radius = cv2.minEnclosingCircle(contour)
                x, y = int(x), int(y)
                INITIAL_SHOTS.append((x, y))
                cv2.circle(captured_frame, (x, y), 5, (0, 0, 255), -1)
            
            print(f"Initial shots detected and saved: {INITIAL_SHOTS}")
            cv2.imshow("Initial Shots Detected", captured_frame)
        elif key == ord('q'):  # Quit
            print("Exiting initial shot detection.")
            break

    cap.release()
    cv2.destroyAllWindows()

def is_new_shot(x, y, tolerance=10):
    """Check if a detected shot is new (not in initial shots)."""
    for shot_x, shot_y in INITIAL_SHOTS:
        if calculate_distance(x, y, shot_x, shot_y) <= tolerance:
            return False
    return True


def preview_and_process():
    """Show real-time preview and process captured frame in the same window."""
    global captured_frame, CENTER_X, CENTER_Y

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

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
                if not filter_contour(contour):  # Use the same filter function
                    continue

                (x, y), radius = cv2.minEnclosingCircle(contour)
                x, y = int(x), int(y)

                # Check if the shot is new
                if not is_new_shot(x, y):
                    continue  # Ignore this shot

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
            # cv2.putText(preview_frame, "Press 'C' to capture", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            # cv2.putText(preview_frame, "Press 'Q' to Quit", (470, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            # cv2.putText(preview_frame, "Press 'R' to Recapture", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(preview_frame, "Press 'C' to capture", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(preview_frame, "Press 'Q' to Quit", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(preview_frame, "Press 'R' to Recapture", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

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

    # Center the window
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    window_width = 400
    window_height = 300
    position_top = int(screen_height / 2 - window_height / 2)
    position_right = int(screen_width / 2 - window_width / 2)
    root.geometry(f'{window_width}x{window_height}+{position_right}+{position_top}')

    root.title("J-Forces")
    root.iconbitmap("./—Pngtree—eagle logo design_5369991.ico")

    # Label for the title
    label = Label(root, text="Shooting Range Scoring System", font=("Helvetica", 16))
    label.pack(pady=10)

    # Load Template Button
    load_template_button = Button(root, text="Load Template", command=load_template, width=20, height=2)
    load_template_button.pack(pady=10)

    # Detect initial shot Button
    initial_shots_button = Button(root, text="Detect Initial Shots", command=detect_initial_shots, width=20, height=2)
    initial_shots_button.pack(pady=10)

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