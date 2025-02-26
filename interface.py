import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyttsx3

# Initialize Text-to-Speech engine
engine = pyttsx3.init()

model = pickle.load(open('./model.pickle', 'rb'))

cap = cv2.VideoCapture(0)  # Initialize video capture from default camera

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Create a DrawingSpec object with the desired style
hand_landmark_style = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
hand_connection_style = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)

labels_dict = {0: 'Hi', 1: 'Sorry', 2: 'Thanks', 3: 'Good', 4: 'Bad', 5: 'Ok!', 6: 'Love u'}

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()  # Read frame from video capture

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                hand_landmark_style,
                hand_connection_style)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x - min(x_))
            data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])

        predicted_label = labels_dict.get(int(prediction[0]), "Unknown Gesture")

        # Speak out the recognized gesture
        engine.say(predicted_label)
        engine.runAndWait()

        # Save the frame as a photo
        cv2.imwrite('output.jpg', frame)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()