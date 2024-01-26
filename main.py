import cv2 as cv
import mediapipe as mp

# right_key and  left_key are  Hexadecimal representation of virtual key codes on keyboard
# corresponds to right arrow and left arrow key respectively.
# PressKey and  ReleaseKey are the functions to press and release key based on Hexadecimal representations as parameter.
from KeyControls import right_key, left_key
from KeyControls import PressKey, ReleaseKey

# creation of video capture object and setting window size.
video = cv.VideoCapture(0)
video.set(3, 640)
video.set(4, 480)

# creation of hand tracking object.
HandTrack = mp.solutions.hands.Hands(min_detection_confidence=0.6, max_num_hands=1)

lmList = [8, 12, 16, 20]

while True:

    # reading video.
    success, frame = video.read()

    # flipping the video making more interactive as it maps the user reaction.
    frame = cv.flip(frame, 1)

    # precessing the image and getting landmarks.
    rgbImage = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    tracking_results = HandTrack.process(image=rgbImage)

    # hand is present follows the code.
    if tracking_results.multi_hand_landmarks:
        h, w, c = frame.shape

        # landMark_list is used to store all the landmark coordinates of a hand.
        # finger_count is used to store number of fingers are opened appends 1 for every opened finger.
        finger_count, landMark_list = [], []

        # getting landmarks of the hand.
        for _, landMark in enumerate(tracking_results.multi_hand_landmarks[0].landmark):
            xc, yc = int(landMark.x * w), int(landMark.y * h)
            landMark_list.append([xc, yc])

        # condition to check if the finger  is opened or not if opened ,appends 1 to finger_count list basing on the flipped image.
        if landMark_list[4][0] < landMark_list[5][0]:
            finger_count.append(1)
        for Id in lmList:
            if landMark_list[Id][1] < landMark_list[Id - 2][1]:
                finger_count.append(1)

        # if every finger is opened count is equals to 5 indicating release of gas else goes with break.
        # if count is 5 the left key  released if it pressed else it harms nothing , followed by pressing right key and same with else block.
        if finger_count.count(1) == 5:
            ReleaseKey(left_key)
            PressKey(right_key)
        else:
            ReleaseKey(right_key)
            PressKey(left_key)
    cv.imshow("video", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
