from flask import Flask, render_template, Response, request, redirect, url_for, flash
import cv2
import numpy as np
from deepface import DeepFace
import json
import os
import csv
from datetime import datetime
from sklearn.preprocessing import Normalizer
from sklearn.metrics.pairwise import cosine_similarity
import time
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Paths and global data
registered_embeddings_path = "registered_embeddings.npy"
registered_labels_path = "registered_labels.json"
attendance_file = "attendance.csv"
upload_folder = "uploads"
os.makedirs(upload_folder, exist_ok=True)

if os.path.exists(registered_embeddings_path) and os.path.exists(registered_labels_path):
    registered_embeddings = np.load(registered_embeddings_path)
    with open(registered_labels_path, "r") as f:
        registered_labels = json.load(f)
else:
    registered_embeddings = np.empty((0, 128))
    registered_labels = {}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

marked_students = set()
pause_start = None
pause_duration = 3
paused_name = None

video_stream_url = "http://192.168.100.65:4747/video"

def save_registered_data():
    global registered_embeddings, registered_labels
    np.save(registered_embeddings_path, registered_embeddings)
    with open(registered_labels_path, "w") as f:
        json.dump(registered_labels, f)

def register_student(name, embedding):
    global registered_embeddings, registered_labels
    normalizer_local = Normalizer(norm="l2")
    embedding_norm = normalizer_local.transform([embedding])[0]

    registered_embeddings = np.vstack([registered_embeddings, embedding_norm])
    new_index = str(len(registered_labels))
    registered_labels[new_index] = name

    save_registered_data()
    print(f"✅ Registered student: {name} at index {new_index}")

def register_from_image(image_path, name):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rep = DeepFace.represent(img_rgb, model_name="Facenet", enforce_detection=False)[0]["embedding"]
    register_student(name, rep)

def detect_face_opencv(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None, None
    x, y, w, h = faces[0]
    return img[y:y+h, x:x+w], (x, y, w, h)

def recognize_student(embedding, threshold=0.7):
    if registered_embeddings.shape[0] == 0:
        return "Unknown", None
    normalizer_local = Normalizer(norm="l2")
    embedding_norm = normalizer_local.transform([embedding])
    similarities = cosine_similarity(embedding_norm, registered_embeddings)[0]
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]
    if best_score >= threshold:
        matched_name = registered_labels.get(str(best_idx), "Unknown")
        return matched_name, best_score
    else:
        return "Unknown", best_score

def mark_attendance(name, subject='General'):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(attendance_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([name, now, subject])

marked_students_per_subject = {}

def generate_frames(subject):
    global pause_start, paused_name

    # Initialize attendance set for this subject if not exists
    if subject not in marked_students_per_subject:
        marked_students_per_subject[subject] = set()

    cap = cv2.VideoCapture(video_stream_url)
    if not cap.isOpened():
        raise RuntimeError("Could not open video stream")

    frame_count = 0
    detection_interval = 10  # Process only every 10th frame
    label = "No Face Detected"
    color = (0, 0, 255)
    last_bbox = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            frame_count += 1

            # Pause display logic: show attendance marked message for a few seconds
            if pause_start is not None:
                elapsed = time.time() - pause_start
                if elapsed >= pause_duration:
                    pause_start = None
                    paused_name = None
                else:
                    msg = f"Attendance marked for {paused_name} ({subject})"
                    cv2.putText(frame, msg, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 0), 3)
                    _, buffer = cv2.imencode('.jpg', frame)
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                           buffer.tobytes() + b'\r\n')
                    continue

            # Perform face detection and recognition every N frames
            if frame_count % detection_interval == 0:
                face, bbox = detect_face_opencv(frame)
                label = "No Face Detected"
                color = (0, 0, 255)
                last_bbox = bbox

                if face is not None:
                    try:
                        # Preprocess face for embedding extraction
                        small_face = cv2.resize(face, (160, 160))
                        face_rgb = cv2.cvtColor(small_face, cv2.COLOR_BGR2RGB)
                        rep = DeepFace.represent(
                            face_rgb,
                            model_name="Facenet",
                            enforce_detection=False
                        )[0]["embedding"]

                        name, score = recognize_student(rep)
                        if name != "Unknown":
                            # Mark attendance if not already marked for this subject
                            if name not in marked_students_per_subject[subject]:
                                mark_attendance(name, subject=subject)
                                marked_students_per_subject[subject].add(name)
                                pause_start = time.time()
                                paused_name = name
                            label = f"{name} ({score:.2f})"
                            color = (0, 255, 0)
                        else:
                            label = "Unknown"
                            color = (0, 0, 255)
                    except Exception:
                        label = "Recognition Error"
                        color = (0, 0, 255)

            # Draw bounding box and label on the frame
            if last_bbox is not None:
                x, y, w, h = last_bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            else:
                # Show label in fixed position if no face detected
                cv2.putText(frame, label, (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Encode frame as JPEG and yield for streaming
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                   buffer.tobytes() + b'\r\n')

            cv2.waitKey(1)

    finally:
        cap.release()

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/admin')
def admin():
    return render_template('admin.html')

@app.route('/student', methods=['GET', 'POST'])
def student():
    attendance_records = []
    attendance_checked = False
    student_name = ""
    subject = ""

    if request.method == 'POST':
        student_name = request.form.get('student_name', '').strip()
        subject = request.form.get('subject', '').strip()
        attendance_checked = True

        if os.path.exists(attendance_file):
            with open(attendance_file, 'r') as f:
                reader = csv.reader(f)
                attendance_records = [row[1] for row in reader
                                      if row and row[0].lower() == student_name.lower() and row[2].lower() == subject.lower()]

    subjects = ['AI', 'DSA', 'DAA', 'MAD']

    return render_template('student.html',
                           attendance_records=attendance_records,
                           attendance_checked=attendance_checked,
                           student_name=student_name,
                           subject=subject,
                           subjects=subjects)
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        if 'image' not in request.files or 'name' not in request.form:
            flash("No image or name provided!")
            return redirect(url_for('register'))

        file = request.files['image']
        name = request.form['name'].strip()

        if file.filename == '' or name == '':
            flash("Please provide both name and image!")
            return redirect(url_for('register'))

        filename = secure_filename(file.filename)
        filepath = os.path.join(upload_folder, filename)
        file.save(filepath)

        try:
            register_from_image(filepath, name)
            flash(f"Student '{name}' registered successfully!")
        except Exception as e:
            flash(f"Failed to register student: {e}")

        os.remove(filepath)
        return redirect(url_for('register'))

    return render_template('register.html')

@app.route('/attendance')
def attendance():
    subjects = ['AI', 'DSA', 'DAA', 'MAD']
    selected_subject = request.args.get('subject', None)

    return render_template('attendance.html',
                           subjects=subjects,
                           subject=selected_subject)

@app.route('/video_feed')
def video_feed():
    subject = request.args.get('subject', 'General')  # fallback to 'General'
    return Response(generate_frames(subject),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
