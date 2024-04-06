import numpy as np
import cv2

# Hold the background frame for background subtraction.
background = None
# Hold the hand's data so all its details are in one place.
hand = None
# Variables to count how many frames have passed and to set the size of the window.
frames_elapsed = 0
FRAME_HEIGHT = 200
FRAME_WIDTH = 300
# Humans come in a ton of beautiful shades and colors.
# Try editing these if your program has trouble recognizing your skin tone.
CALIBRATION_TIME = 30
BG_WEIGHT = 0.5
OBJ_THRESHOLD = 18

# Our region of interest will be the top right part of the frame.
region_top = 0
region_bottom = int(2 * FRAME_HEIGHT / 3)
region_left = int(FRAME_WIDTH / 2)
region_right = FRAME_WIDTH

frames_elapsed = 0

capture = cv2.VideoCapture(0)

class HandData:
    def __init__(self, top, bottom, left, right, centerX):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        self.centerX = centerX
        self.prevCenterX = 0
        self.isInFrame = False
        self.isWaving = False
        self.fingers = None

    def update(self, top, bottom, left, right):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

# Function to calculate the centroid of a hand.
def calculate_centroid(hand):
    centroid_x = (hand.left + hand.right) // 2
    centroid_y = (hand.top + hand.bottom) // 2
    return centroid_x, centroid_y

# Define the missing function for writing on the image.
def write_on_image(frame):
    text = "Searching..."

    if frames_elapsed < CALIBRATION_TIME:
        text = "Calibrating..."
    elif hand is None or not hand.isInFrame:
        text = "No hand detected"
    else:
        if hand.isWaving:
            text = "Waving"
        else:
            # Count fingers based on convexity defects.
            defects = cv2.convexityDefects(hand.contour, hand.hull)
            finger_count = 0

            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(hand.contour[s][0])
                    end = tuple(hand.contour[e][0])
                    far = tuple(hand.contour[f][0])

                    # Calculate triangle area using Heron's formula.
                    a = np.linalg.norm(np.array(far) - np.array(start))
                    b = np.linalg.norm(np.array(far) - np.array(end))
                    c = np.linalg.norm(np.array(start) - np.array(end))
                    area = np.sqrt((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c)) / 4

                    # Apply the angle and area thresholds to identify fingertips.
                    angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c))
                    angle_threshold = np.pi / 2  # Adjust this value based on your specific conditions.
                    area_threshold = 4000  # Adjust this value based on your specific conditions.

                    if angle < angle_threshold and area > area_threshold:
                        finger_count += 1

                text = f"Fingers: {finger_count}"

    cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # Highlight the region of interest.
    cv2.rectangle(frame, (region_left, region_top), (region_right, region_bottom), (255, 255, 255), 2)

while True:
    # Store the frame from the video capture and resize it to the window size.
    ret, frame = capture.read()
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    # Flip the frame over the vertical axis so that it works like a mirror, which is more intuitive to the user.
    frame = cv2.flip(frame, 1)

    # Convert the frame to grayscale for background subtraction.
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if frames_elapsed < CALIBRATION_TIME:
        # Calibrate the background during the first few frames.
        background = gray_frame.copy().astype(float)
    else:
        # Subtract the background to detect the hand.
        diff = cv2.absdiff(background.astype(np.uint8), gray_frame)
        _, thresholded = cv2.threshold(diff, OBJ_THRESHOLD, 255, cv2.THRESH_BINARY)

        # Find contours in the thresholded image.
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize hand object.
        hand = HandData(0, 0, 0, 0, 0)

        # Loop over the contours to find the hand.
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Adjust this value based on your specific conditions.
                hand.contour = contour
                epsilon = 0.03 * cv2.arcLength(contour, True)
                hand.approx = cv2.approxPolyDP(contour, epsilon, True)
                hand.hull = cv2.convexHull(contour)
                x, y, w, h = cv2.boundingRect(contour)
                hand.update(y, y + h, x, x + w)

                # Draw a bounding box around the hand.
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Calculate centroid of the hand.
                hand.centerX, _ = calculate_centroid(hand)

                # Perform finger counting.
                write_on_image(frame)

    # Show the previously captured frame.
    cv2.imshow("Camera Input", frame)
    frames_elapsed += 1

    # Check if user wants to exit.
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

# When we exit the loop, we have to stop the capture too.
capture.release()
cv2.destroyAllWindows()
