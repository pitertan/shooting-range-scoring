import cv2
import math
import tkinter as tk
import os
from tkinter import Label, Button, filedialog, ttk
from PIL import Image, ImageTk
import numpy as np

# Global variables
CENTER_X, CENTER_Y = 0, 0
RADIUS_SCORES = []
SCORES = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
INITIAL_SHOTS = []  # List untuk menyimpan koordinat tembakan awal
template_image = None
captured_frame = None  # Frame terakhir yang di-capture
cam_index = 0  # Initialize default camera index
base_path = "D:/J-forces Project/shooting-range-scoring"

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

def filter_contour(contour, image):
    """Apply filtering criteria to a contour and return True if it's valid."""
    contour_area = cv2.contourArea(contour)
    if contour_area < 20 or contour_area > 2000:  # Filter based on area, default 1000
        return False

    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    if aspect_ratio < 0.1 or aspect_ratio > 2:  # Filter based on aspect ratio default 0.5 | 1.5
        return False

    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * math.pi * contour_area / (perimeter ** 2)
    if circularity < 0.1:  # Filter based on circularity
        return False
    
    # Ambil ROI dalam bounding box
    roi = image[y:y+h, x:x+w]

    # Konversi ke Grayscale
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Hitung rata-rata intensitas warna dalam ROI
    mean_intensity = np.mean(gray_roi)

    # Filter berdasarkan warna (bekas tembakan harus hitam)
    if mean_intensity > 60:  # 0 = hitam, 255 = putih (Threshold bisa disesuaikan)
        return False

    return True


def detect_initial_shots():
    """Detect initial shots on the target and store their coordinates."""
    global INITIAL_SHOTS, captured_frame, CENTER_X, CENTER_Y, cam_index

    print(f"Opening camera {cam_index} for initial shot detection")
    cap = cv2.VideoCapture(cam_index)
    
    if not cap.isOpened():
        print(f"Failed to open camera {cam_index}")
        return
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Detecting initial shots. Press 'C' to capture and save initial state. Press 'Q' to quit.")
    
    INITIAL_SHOTS.clear()

    # Variables for animated circle radius
    radius_min = 3
    radius_max = 15
    radius_increment = 1
    current_radius = radius_min

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break
        captured_frame = frame.copy()

        # Get frame dimensions
        height, width, _ = captured_frame.shape
        CENTER_X, CENTER_Y = width // 2, height // 2

        # Draw concentric circles for scoring zones
        max_radius = min(width, height) // 2
        for radius in RADIUS_SCORES:
            if radius > max_radius:
                radius = max_radius
            cv2.circle(captured_frame, (CENTER_X, CENTER_Y), radius, (255, 0, 0), 1)

        # Draw target center
        cv2.circle(captured_frame, (CENTER_X, CENTER_Y), 5, (0, 0, 255), -1)
        cv2.putText(captured_frame, "Target", (CENTER_X + 10, CENTER_Y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw animated circles around initial shots
        for (x, y) in INITIAL_SHOTS:
            cv2.circle(captured_frame, (x, y), 5, (0, 255, 255), -1) 
            cv2.circle(captured_frame, (x, y), current_radius, (0, 0, 255), 2)  

        # Update radius for animation
        current_radius += radius_increment
        if current_radius >= radius_max or current_radius <= radius_min:
            radius_increment *= -1

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
                if not filter_contour(contour, captured_frame):
                    continue

                (x, y), radius = cv2.minEnclosingCircle(contour)
                x, y = int(x), int(y)
                INITIAL_SHOTS.append((x, y))

            print(f"Initial shots detected and saved: {INITIAL_SHOTS}")
        elif key == ord('q'):  # Quit
            print("Exiting initial shot detection.")
            break

    cap.release()
    cv2.destroyAllWindows()

# Fungsi untuk mendeteksi kamera yang tersedia
def get_available_cameras():
    available_cameras = []
    for i in range(5):  # Try detecting up to 10 cameras
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()  # Try to read a frame
            if ret:  # Only add camera if it can actually capture frames
                available_cameras.append(i)
            cap.release()
    return available_cameras if available_cameras else [0]  # Return [0] if no cameras found

# Fungsi untuk memilih kamera
def select_camera():
    global cam_index
    
    def set_camera():
        global cam_index
        selected_cam = int(camera_var.get())
        # Test if the selected camera works
        cap = cv2.VideoCapture(selected_cam)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                cam_index = selected_cam
                print(f"Successfully selected camera {cam_index}")
            else:
                print(f"Failed to read from camera {selected_cam}")
        cap.release()
        camera_window.destroy()
    
    camera_window = tk.Toplevel()
    camera_window.title("Select Camera")
    camera_window.geometry("300x150")
    
    available_cameras = get_available_cameras()
    
    ttk.Label(camera_window, text="Select Camera:").pack(pady=10)
    
    camera_var = tk.StringVar(value=str(cam_index))  # Set current camera as default
    camera_dropdown = ttk.Combobox(camera_window, textvariable=camera_var, 
                                 values=[str(cam) for cam in available_cameras])
    camera_dropdown.pack(pady=5)
    
    # Add a test button
    def test_camera():
        try:
            test_cam = int(camera_var.get())
            cap = cv2.VideoCapture(test_cam)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    cv2.imshow(f"Testing Camera {test_cam}", frame)
                    cv2.waitKey(2000)  # Show for 2 seconds
                    cv2.destroyAllWindows()
                    print(f"Successfully tested camera {test_cam}")
                else:
                    print(f"Failed to read from camera {test_cam}")
            cap.release()
        except Exception as e:
            print(f"Error testing camera: {e}")
    
    ttk.Button(camera_window, text="Test Camera", command=test_camera).pack(pady=5)
    ttk.Button(camera_window, text="Select", command=set_camera).pack(pady=5)
    
    # Make the camera selection window modal
    camera_window.transient(camera_window.master)
    camera_window.grab_set()
    camera_window.wait_window()


def is_new_shot(x, y, tolerance=10):
    """Check if a detected shot is new (not in initial shots)."""
    for shot_x, shot_y in INITIAL_SHOTS:
        if calculate_distance(x, y, shot_x, shot_y) <= tolerance:
            return False
    return True


def preview_and_process():
    """Show real-time preview and process captured frame in the same window."""
    global captured_frame, CENTER_X, CENTER_Y, cam_index

    print(f"Opening camera {cam_index} for preview and processing")
    cap = cv2.VideoCapture(cam_index)
    
    if not cap.isOpened():
        print(f"Failed to open camera {cam_index}")
        return
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    process_mode = False  # Flag to toggle between preview and processing

    while True:
        if not process_mode:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
                break
            captured_frame = frame.copy()

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
                if not filter_contour(contour, captured_frame):
                    continue

                (x, y), radius = cv2.minEnclosingCircle(contour)
                x, y = int(x), int(y)

                # Check if the shot is new
                if not is_new_shot(x, y):
                    continue

                score = calculate_score(x, y)
                scores.append(score)

                # Draw detected dot and score
                cv2.circle(result_frame, (x, y), 5, (0, 0, 255), -1)
                cv2.line(result_frame, (CENTER_X, CENTER_Y), (x, y), (0, 255, 255), 2)
                cv2.putText(result_frame, f"{score}", (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            total_score = sum(scores)
            cv2.putText(result_frame, f"Total Score: {total_score}", (10, result_frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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
            cv2.putText(preview_frame, "Press 'Q' to Quit", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(preview_frame, "Press 'R' to Recapture", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(preview_frame, f"Camera Index: {cam_index}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            cv2.imshow("Shooting Range", preview_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            print("Frame captured. Processing...")
            process_mode = True
        elif key == ord('r'):
            print("Returning to preview mode.")
            process_mode = False
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

def goodbye_screen():
    goodbye = tk.Tk()

    # Center the window
    screen_width = goodbye.winfo_screenwidth()
    screen_height = goodbye.winfo_screenheight()

    window_width = 400
    window_height = 200
    position_top = int(screen_height / 2 - window_height / 2)
    position_right = int(screen_width / 2 - window_width / 2)
    goodbye.geometry(f'{window_width}x{window_height}+{position_right}+{position_top}')

    goodbye.title("Full Random Technology")
    # Gunakan path absolut untuk menghindari masalah path relatif
    icon_path = os.path.abspath(os.path.join(base_path, "images", "logo.ico"))
    
    # Panggil iconbitmap dengan path absolut
    goodbye.iconbitmap(icon_path)

    # Load background image
    try:
        bg_image = Image.open(os.path.join(base_path, "images", "background.jpg"))  # Replace with your background image path
        bg_photo = ImageTk.PhotoImage(bg_image)

        # Create canvas to hold the background image
        canvas = tk.Canvas(goodbye, width=screen_width, height=screen_height)
        canvas.place(x=0, y=0)  # Place canvas at the top-left corner of the window

        # Display the background image on canvas
        canvas.create_image(0, 0, image=bg_photo, anchor="nw")

        # Keep a reference to the image to prevent garbage collection
        canvas.image = bg_photo
    except Exception as e:
        print("Error loading background image:", e)

    # Label for goodbye message
    label = Label(goodbye, text="Thank you for using the system!\nGoodbye!", font=("Helvetica", 16))
    label.place(relx=0.5, rely=0.5, anchor="center")  # Centered vertically and horizontally

    # Set a timer to close the goodbye screen after a few seconds
    goodbye.after(1500, lambda: goodbye.destroy())  # Close after 3 seconds

    # Start the goodbye screen
    goodbye.mainloop()

def main_gui():
    root = tk.Tk()

    # Center the window
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    window_width = 400
    window_height = 380
    position_top = int(screen_height / 2 - window_height / 2)
    position_right = int(screen_width / 2 - window_width / 2)
    root.geometry(f'{window_width}x{window_height}+{position_right}+{position_top}')

    root.title("Full Random Technology")
    # Gunakan path absolut untuk menghindari masalah path relatif
    icon_path = os.path.abspath(os.path.join(base_path, "images", "logo.ico"))
    
    # Panggil iconbitmap dengan path absolut
    root.iconbitmap(icon_path)

    # Load background image
    try:
        bg_image = Image.open(os.path.join(base_path, "images", "background.jpg"))  # Replace with your background image path
        bg_photo = ImageTk.PhotoImage(bg_image)

        # Create canvas to hold the background image
        canvas = tk.Canvas(root, width=screen_width, height=screen_height)
        canvas.place(x=0, y=0)  # Place canvas at the top-left corner of the window

        # Display the background image on canvas
        canvas.create_image(0, 0, image=bg_photo, anchor="nw")

        # Keep a reference to the image to prevent garbage collection
        canvas.image = bg_photo
    except Exception as e:
        print("Error loading background image:", e)

    # Label for the title
    label = Label(root, text="Shooting Range Scoring System", font=("Helvetica", 16))
    label.pack(pady=10)

    # Load Template Button
    load_template_button = Button(root, text="Load Template", command=load_template, width=20, height=2)
    load_template_button.pack(pady=10)

    # select camera
    load_template_button = Button(root, text="Select Camera", command=select_camera, width=20, height=2)
    load_template_button.pack(pady=10)

    # Detect initial shot Button
    initial_shots_button = Button(root, text="Detect Initial Shots", command=detect_initial_shots, width=20, height=2)
    initial_shots_button.pack(pady=10)

    # Capture Frame Button
    capture_button = Button(root, text="Start Detection", command=preview_and_process, width=20, height=2)
    capture_button.pack(pady=10)

    # Exit Button
    exit_button = Button(root, text="Exit", command=lambda: [root.destroy(), goodbye_screen()], width=20, height=2, bg="#f54242", fg="white")
    exit_button.pack(pady=10)

    # Start GUI
    root.mainloop()

def welcome_screen():
    welcome = tk.Tk()

    # Center the window
    screen_width = welcome.winfo_screenwidth()
    screen_height = welcome.winfo_screenheight()

    window_width = 600
    window_height = 400
    position_top = int(screen_height / 2 - window_height / 2)
    position_right = int(screen_width / 2 - window_width / 2)
    welcome.geometry(f'{window_width}x{window_height}+{position_right}+{position_top}')

    welcome.title("Full Random Technology")

    # Gunakan path absolut untuk menghindari masalah path relatif
    icon_path = os.path.abspath(os.path.join(base_path, "images", "logo.ico"))
    
    # Panggil iconbitmap dengan path absolut
    welcome.iconbitmap(icon_path)

    # Load background image
    try:
        bg_image = Image.open(os.path.join(base_path, "images", "background.jpg"))  # Replace with your background image path
        bg_image = bg_image.resize((window_width, window_height), Image.Resampling.LANCZOS)
        bg_photo = ImageTk.PhotoImage(bg_image)

        # Create canvas to hold the background image
        canvas = tk.Canvas(welcome, width=screen_width, height=screen_height)
        canvas.place(x=0, y=0)  # Place canvas at the top-left corner of the window

        # Display the background image on canvas
        canvas.create_image(0, 0, image=bg_photo, anchor="nw")

        # Keep a reference to the image to prevent garbage collection
        canvas.image = bg_photo
    except Exception as e:
        print("Error loading background image:", e)

    # Load and display the logo
    try:
        logo_image = Image.open(os.path.join(base_path, "images", "main-logo.png"))  # Replace with your logo file path
        logo_image = logo_image.resize((200, 200), Image.Resampling.LANCZOS)
        logo_photo = ImageTk.PhotoImage(logo_image)
        logo_label = Label(welcome, image=logo_photo, bg="white")
        logo_label.image = logo_photo
        logo_label.place(relx=0.5, rely=0.3, anchor="center")  # Logo positioned near the top
    except Exception as e:
        print("Error loading logo:", e)

    # Title Label
    title_label = Label(welcome, text="Welcome to Shooting Range Scoring System", font=("Helvetica", 16), bg="white")
    title_label.place(relx=0.5, rely=0.65, anchor="center")  # Positioned below the logo

    # Start Button
    start_button = Button(welcome, text="Start", command=lambda: [welcome.destroy(), main_gui()], width=20, height=2)
    start_button.place(relx=0.5, rely=0.8, anchor="center")  # Positioned below the title label

    # Run the welcome screen
    welcome.mainloop()

if __name__ == "__main__":
    welcome_screen()