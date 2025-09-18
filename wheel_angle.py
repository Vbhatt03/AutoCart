import cv2
import numpy as np
import math
import time

CAM_INDEX = 0
MAX_ANGLE_ABS = 420.0

# ArUco setup
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

def find_wheel_center(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9,9), 2)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                               param1=50, param2=30, minRadius=50, maxRadius=500)
    if circles is None:
        return None
    circles = np.round(circles[0, :]).astype("int")
    circles = sorted(circles, key=lambda c: c[2], reverse=True)
    x,y,r = circles[0]
    return (x,y,r)

def angle_from_center(center, pt):
    dx = pt[0] - center[0]
    dy = center[1] - pt[1]  # invert y-axis so up is positive
    ang_rad = math.atan2(dy, dx)
    ang_deg = math.degrees(ang_rad)
    # 0° is up
    angle_from_up = 90 - ang_deg
    return ((angle_from_up + 180) % 360) - 180

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    time.sleep(0.5)
    ret, frame = cap.read()
    if not ret:
        print("Camera failed.")
        return

    found = find_wheel_center(frame)
    if found:
        x,y,r = found
        center = (x,y)
        print("Wheel center:", center, "r:", r)
    else:
        print("No circle found. Click center manually.")
        center = None
        clicked = []
        def on_mouse(ev, x, y, flags, param):
            if ev == cv2.EVENT_LBUTTONDOWN:
                clicked.append((x,y))
        cv2.namedWindow("pick")
        cv2.setMouseCallback("pick", on_mouse)
        while True:
            cv2.imshow("pick", frame)
            k = cv2.waitKey(1) & 0xFF
            if clicked:
                center = clicked[-1]
                print("Picked center:", center)
                break
            if k == 13:
                break
        cv2.destroyWindow("pick")
        if center is None:
            return

    prev_angle = None
    rotation_offset = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        corners, ids, _ = detector.detectMarkers(frame)

        if ids is not None and len(corners) > 0:
            # Use first detected tag center
            pts = corners[0][0]
            tag_center = np.mean(pts, axis=0).astype(int)
            raw_angle = angle_from_center(center, tag_center)

            if prev_angle is None:
                prev_angle = raw_angle

            delta = raw_angle - prev_angle
            if delta > 180: rotation_offset -= 360
            elif delta < -180: rotation_offset += 360
            continuous_angle = raw_angle + rotation_offset
            prev_angle = raw_angle

            clamped = max(-MAX_ANGLE_ABS, min(MAX_ANGLE_ABS, continuous_angle))

            # Draw for debug
            cv2.circle(frame, center, 3, (255,255,0), -1)
            cv2.circle(frame, tuple(tag_center), 4, (0,255,0), -1)
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            txt = f"Angle: {continuous_angle:.2f}  Clamped: {clamped:.2f}"
            cv2.putText(frame, txt, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            print(f"{time.time():.2f}\t{continuous_angle:.2f}")

        else:
            cv2.putText(frame, "Tag not found", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.imshow("frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
