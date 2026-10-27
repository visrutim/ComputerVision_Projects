
import cv2
import mediapipe as mp
import pyautogui
import time

# =====================================
# 1. INITIALIZE MEDIAPIPE
# =====================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# =====================================
# 2. OPEN WEBCAM
# =====================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    hands.close()
    raise SystemExit

# =====================================
# 3. SCROLLING SETTINGS
# =====================================

SCROLL_SPEED = 15
SCROLL_INTERVAL = 0.01
GESTURE_HOLD_TIME = 0.10

scroll_mode = "STOP"

candidate_gesture = "NONE"
candidate_since = time.time()
last_processed_gesture = "NONE"

last_scroll_time = 0


# =====================================
# 4. DETECT HAND GESTURE
# =====================================

def detect_gesture(lm):

    # Check if fingers are folded
    index_folded = lm[8].y > lm[6].y
    middle_folded = lm[12].y > lm[10].y
    ring_folded = lm[16].y > lm[14].y
    pinky_folded = lm[20].y > lm[18].y

    # Index finger only = Scroll up
    if (
        not index_folded
        and middle_folded
        and ring_folded
        and pinky_folded
    ):
        return "INDEX"

    # Pinky finger only = Scroll down
    if (
        index_folded
        and middle_folded
        and ring_folded
        and not pinky_folded
    ):
        return "PINKY"

    # No recognized gesture
    return "NONE"


# =====================================
# 5. MAIN PROGRAM
# =====================================

print("\nHand Gesture Scrolling Started!")
print("Index finger = Toggle scrolling UP")
print("Pinky finger = Toggle scrolling DOWN")
print("Show the same gesture again = STOP")
print("Press Q to quit\n")

try:
    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            print("Error: Cannot read webcam frame.")
            break

        # Mirror webcam image
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = hands.process(rgb_frame)

        detected_gesture = "NONE"

        # ---------------------------------
        # Detect and draw hand landmarks
        # ---------------------------------

        if results.multi_hand_landmarks:

            hand_landmarks = results.multi_hand_landmarks[0]
            lm = hand_landmarks.landmark

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            detected_gesture = detect_gesture(lm)

        now = time.time()

        # ---------------------------------
        # Stabilize gesture detection
        # ---------------------------------

        if detected_gesture != candidate_gesture:
            candidate_gesture = detected_gesture
            candidate_since = now

        stable = (
            now - candidate_since >= GESTURE_HOLD_TIME
        )

        # ---------------------------------
        # Toggle scrolling mode
        # ---------------------------------

        if (
            stable
            and detected_gesture in ("INDEX", "PINKY")
            and detected_gesture != last_processed_gesture
        ):

            if detected_gesture == "INDEX":

                if scroll_mode == "SCROLL_UP":
                    scroll_mode = "STOP"
                else:
                    scroll_mode = "SCROLL_UP"

            elif detected_gesture == "PINKY":

                if scroll_mode == "SCROLL_DOWN":
                    scroll_mode = "STOP"
                else:
                    scroll_mode = "SCROLL_DOWN"

            last_processed_gesture = detected_gesture

        # Allow a new gesture after returning to neutral
        if detected_gesture == "NONE":
            last_processed_gesture = "NONE"

        # ---------------------------------
        # Continuous scrolling
        # ---------------------------------

        if now - last_scroll_time >= SCROLL_INTERVAL:

            if scroll_mode == "SCROLL_DOWN":
                pyautogui.scroll(-SCROLL_SPEED)
                last_scroll_time = now

            elif scroll_mode == "SCROLL_UP":
                pyautogui.scroll(SCROLL_SPEED)
                last_scroll_time = now

        # ---------------------------------
        # Display status on webcam
        # ---------------------------------

        cv2.putText(
            frame,
            f"Mode: {scroll_mode}",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0) if scroll_mode != "STOP"
            else (0, 0, 0),
            2
        )

        cv2.putText(
            frame,
            "Index: Toggle UP | Pinky: Toggle DOWN",
            (10, 70),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.5,
            (0, 20, 0),
            2
        )

        cv2.putText(
            frame,
            "Repeat gesture: STOP | Q: Quit",
            (10, 105),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.55,
            (0, 0, 0),
            2
        )

        cv2.imshow(
            "Hand Gesture Scrolling",
            frame
        )

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:

    cap.release()
    hands.close()
    cv2.destroyAllWindows()

    print("Hand Gesture Scrolling Stopped.")

