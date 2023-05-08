from flask import Flask, render_template, request, redirect
import os
import numpy as np
import pandas as pd
import cv2
from werkzeug.utils import secure_filename
from tensorflow import keras
import tensorflow as tf
import base64

app = Flask(__name__, template_folder='templates')
UPLOAD_FOLDER = os.path.abspath("uploads/")
ALLOWED_EXTENSIONS = set(['mp4', 'avi', 'mpeg'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def index():
    return render_template("index.html")


def home():
    return render_template("index.html")

@app.route('/predict')
def predict():
    return render_template("predict.html")

@app.route('/adopt')
def adopt():
    return render_template("adopt.html")

@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/form_page')
def form_page():
    return render_template("form_page.html")


@app.route("/animal_list")
def animal_list():
    return render_template("animalList.html")

@app.route("/policy")
def policy():
    return render_template("Policy.html")

@app.route("/success_stories")
def success_stories():
    return render_template("Success.html")

@app.route("/resources")
def resources():
    return render_template("Resources.html")



IMG_SIZE = 224
BATCH_SIZE = 64

MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048


def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y: start_y + min_dim, start_x: start_x + min_dim]


def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)


# Feature Extraction
def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights='imagenet',
        include_top=False,
        pooling='avg',
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name='feature_extractor')


feature_extractor = build_feature_extractor()

train_df = pd.read_csv('train.csv')

label_processor = keras.layers.StringLookup(num_oov_indices=1, vocabulary=np.unique(train_df['tag']))
print(label_processor.get_vocabulary())


def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    return frame_features, frame_mask


Rabies = ['Paralysis', 'Dropped Jaw & Tongue', 'Hyper Salivation', 'Incoordination', 'Seizure']
Normal = ['Barking', 'Running', 'Playing', 'Wagging Tail', 'Digging']
Rab = 0
Nor = 0
No_Detect = 0


def Video_play(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video stream or file.")
        return

    count = 0
    display_count = 0
    processed_frames = []  # Store processed frames

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        count += 1
        face_detector = cv2.CascadeClassifier('cascade.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in num_faces:
            if Rab > Nor:
                cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 0, 255), 4)
                cv2.putText(frame, "Rabies", (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_4)
            elif Nor > Rab:
                cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
                cv2.putText(frame, "Normal", (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_4)
            else:
                cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 4)

        if count <= 3:
            processed_frames.append(frame)
            display_count += 1

        if display_count == 3:
            break

    cap.release()
    # cv2.destroyAllWindows()

    for i, frame in enumerate(processed_frames):
        cv2.imwrite(f'uploads/processed_frame_{i+1}.jpg', frame)

    print(f"Total frames processed: {count}")
    print(f"Total frames displayed: {display_count}")
    return processed_frames


@app.route("/upload", methods=["POST"])
def upload():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Load the model from the saved files
            with open('model.json', 'r') as f:
                model_json = f.read()
            sequence_model = tf.keras.models.model_from_json(model_json)
            sequence_model.load_weights('model.h5')

            global Rab
            global Nor
            global No_Detect
            Rab = 0
            Nor = 0
            No_Detect = 0
            count = 0
            class_vocab = label_processor.get_vocabulary()

            processed_frames = Video_play(path)
            processed_frames_base64 = []
            frames = load_video(path)  # You need to define or import this function
            frame_features, frame_mask = prepare_single_video(frames)  # You need to define or import this function
            probabilities = sequence_model.predict([frame_features, frame_mask])[0]
            for i, frame in enumerate(processed_frames):
                frame_np = np.array(frame)
                _, buffer = cv2.imencode('.jpg', frame_np)
                processed_frames_base64.append(base64.b64encode(buffer).decode())


            print('Details:')
            for i in np.argsort(probabilities)[::-1]:
                if count < len(Rabies):
                    if class_vocab[i] == 'No Detection':
                        No_Detect += 1
                    print(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")
                    if class_vocab[i] in Rabies:
                        Rab += 1
                    else:
                        Nor += 1
                count += 1

            if Rab > Nor:
                result = "Since features of rabies are more dominating among the features shown, the final result can be assumed as RABIES."
            elif Nor > Rab:
                result = "Since features of normal are more dominating among the features shown, the final result can be assumed as NORMAL."
            else:
                result = "No dog detected."

            return render_template("result.html", result=result, processed_frames_base64=processed_frames_base64)

    return redirect("/")


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == "__main__":
    app.run(debug=True)

