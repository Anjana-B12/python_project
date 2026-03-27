import cv2
import numpy as np

# ==========================
# Initialize webcam
# ==========================
cap = cv2.VideoCapture(0)  # 0 = default webcam

# ==========================
# Counter variables
# ==========================
count = 0                # Total objects counted
object_crossed = False   # Flag to track if an object has crossed the line

# ==========================
# Horizontal line position (y-coordinate)
# ==========================
line_y = 250  # Adjust based on webcam feed size

# ==========================
# Background subtractor for motion detection
# ==========================
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the frame for easier interaction
    frame = cv2.flip(frame, 1)

    # ==========================
    # Apply background subtraction to detect motion
    # ==========================
    fgmask = fgbg.apply(frame)

    # ==========================
    # Remove small noise from mask
    # ==========================
    kernel = np.ones((5,5), np.uint8)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    # ==========================
    # Find contours of moving objects
    # ==========================
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected = False  # Flag: True if object is near the line

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 2000:  # Filter out small movements/noise
            x, y, w, h = cv2.boundingRect(cnt)
            cx = x + w // 2  # center x
            cy = y + h // 2  # center y

            # Draw bounding box around detected object
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

            # ==========================
            # Trigger logic
            # Check if object's center is near the horizontal line
            # ==========================
            if abs(cy - line_y) < 10:
                detected = True

    # ==========================
    # Counting logic
    # - Increment counter when object crosses line
    # - Use flag to avoid double-counting
    # ==========================
    if detected and not object_crossed:
        object_crossed = True
        count += 1  # Increase count when object crosses line
    elif not detected and object_crossed:
        object_crossed = False  # Reset flag when object moves away

    # Draw horizontal counting line
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (255,0,0), 2)

    # Display object count on screen
    cv2.putText(frame, f"Objects Counted: {count}",
                (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    # Show live webcam feed
    cv2.imshow("Motion Counter", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup resources
cap.release()
cv2.destroyAllWindows()