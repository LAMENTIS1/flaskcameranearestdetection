from flask import Flask, Response, render_template
import cv2
import mediapipe as mp
import random

app = Flask(__name__)

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

@app.route('/')
def index():
    return render_template('input.html')

def generate_frames():
    with mp_face_mesh.FaceMesh(
        max_num_faces=5,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame color to RGB as required by MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb_frame)

            closest_face_index = -1
            closest_distance = float('inf')

            if result.multi_face_landmarks:
                for i, face_landmarks in enumerate(result.multi_face_landmarks):
                    nose_landmark = face_landmarks.landmark[1]
                    distance = nose_landmark.z

                    if distance < closest_distance:
                        closest_distance = distance
                        closest_face_index = i

                    num_landmarks = len(face_landmarks.landmark)
                    visible_landmarks = random.sample(range(num_landmarks), int(0.60 * num_landmarks))

                    for idx in visible_landmarks:
                        landmark = face_landmarks.landmark[idx]
                        h, w, _ = frame.shape
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        color = (255, 0, 0)

                        if i == closest_face_index:
                            color = (0, 255, 0)

                        cv2.circle(frame, (x, y), 2, color, -1)

            # Encode the frame in JPEG format
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
