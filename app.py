from flask import Flask, render_template, request, jsonify
import cv2
import face_recognition
import numpy as np
import csv
from datetime import datetime, timedelta

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index_face.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Read the image file
    image = face_recognition.load_image_file(file)

    # Face recognition code
    ravi_image = face_recognition.load_image_file("photos/ravi.jpg")
    ravi_encoding = face_recognition.face_encodings(ravi_image)[0]
    kalam_image = face_recognition.load_image_file("photos/kalam.jpg")
    kalam_encoding = face_recognition.face_encodings(kalam_image)[0]
    ratan_image = face_recognition.load_image_file("photos/ratantata.jpg")
    ratan_encoding = face_recognition.face_encodings(ratan_image)[0]
    harsha_image = face_recognition.load_image_file("photos/harsha.jpg")
    harsha_encoding = face_recognition.face_encodings(harsha_image)[0]
    bhanu_image = face_recognition.load_image_file("photos/bhanu.jpg")
    bhanu_encoding = face_recognition.face_encodings(bhanu_image)[0]

    # Storing the encoding and assign names to the encodings
    known_face_encoding = [ravi_encoding, kalam_encoding, ratan_encoding, bhanu_encoding, harsha_encoding]
    known_face_names = ["bhanu", "ravi", "kalam", "ratan", "harsha"]

    # Make a copy of face encodings
    persons = known_face_names.copy()
    print(persons)

    face_locations = []
    face_encodings = []
    face_names = []
    s = True

    # Getting the current date and time
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")

    # Creating a CSV file with the name as the current date
    csv_file_path = current_date + '.csv'
    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Timestamp', 'Location'])

    # Capture single frame after 10 seconds
    start_time = datetime.now()
    capture_interval = timedelta(seconds=10)
    capture_image = False

    video_capture = cv2.VideoCapture(0)

    while (datetime.now() - start_time) < capture_interval:
        _, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Find face locations in the frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []

        for face_encoding in face_encodings:
            # Compare the detected face with known faces
            matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
            name = "Unknown"  # Default name if no match is found

            # Check if any known face matches the detected face
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                current_time = now.strftime("%H-%M-%S")
                csv_writer.writerow([name, current_time])
                print("Date and time captured")

            face_names.append(name)

        # Draw bounding boxes and display names
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Display the name below the face
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            image_filename = 'captured_images/{}.jpg'.format(datetime.now().strftime("%Y%m%d%H%M%S"))
            cv2.imwrite(image_filename, frame)
            print("Image captured and saved.")

        cv2.imshow("face_recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture
    video_capture.release()
    cv2.destroyAllWindows()

    # For demonstration, let's just mark faces in the image
    face_locations = face_recognition.face_locations(image)
    for face_location in face_locations:
        top, right, bottom, left = face_location
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

    # Save the marked image
    cv2.imwrite('static/result.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    return jsonify({'result': 'success'})

if __name__ == '__main__':
    app.run(debug=True)
