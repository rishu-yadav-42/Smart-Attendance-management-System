from collections import Counter, defaultdict
from datetime import datetime
import csv
import os
import re
import sqlite3
import sys
import threading
import time

VENDOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".vendor312")
if os.path.isdir(VENDOR_DIR) and VENDOR_DIR not in sys.path:
    sys.path.insert(0, VENDOR_DIR)

import cv2
import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, redirect, render_template, request, session, url_for
from PIL import Image

try:
    from insightface.app import FaceAnalysis
except ImportError:
    FaceAnalysis = None


app = Flask(__name__)
app.secret_key = "secret123"
APP_VERSION = "arcface-guided-register-v5-2026-04-26"

STUDENT_FILE = "StudentDetails/StudentDetails.csv"
DB_FILE = "attendance.db"
TRAINING_DIR = "TrainingImage"
MODEL_FILE = "TrainingImageLabel/Trainner.yml"
FALLBACK_MODEL_FILE = "TrainingImageLabel/face_model.npz"
ARCFACE_MODEL_FILE = "TrainingImageLabel/arcface_embeddings.npz"
ATTENDANCE_DIR = "Attendance"
ARCFACE_MODEL_ROOT = "FaceModelStore"
ARCFACE_MODEL_NAME = "buffalo_l"
SAMPLE_COUNT = 60
FACE_SIZE = (220, 220)
FACE_DETECT_SCALE_FACTOR = 1.06
FACE_DETECT_MIN_NEIGHBORS = 3
FACE_DETECT_MIN_SIZE = (48, 48)
CONFIDENCE_LIMIT = 50
AVG_CONFIDENCE_LIMIT = 48
FALLBACK_CONFIDENCE_LIMIT = 80
FALLBACK_AVG_CONFIDENCE_LIMIT = 75
ARCFACE_SIMILARITY_THRESHOLD = 0.35
ARCFACE_SECOND_BEST_MARGIN = 0.03
MIN_CONFIDENT_FRAMES = 3
MIN_WIN_RATIO = 0.55
FAST_MATCH_TARGET = 4
REMOVED_STUDENT_IDS = set()
REMOVED_FACE_CONFIDENCE_LIMIT = 48
REMOVED_FACE_FALLBACK_CONFIDENCE_LIMIT = 35
REMOVED_FACE_ARCFACE_THRESHOLD = 0.38
REMOVED_FACE_MIN_FRAMES = 3
CAMERA_FRAME_RETRY_LIMIT = 20

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

arcface_app = None
arcface_init_error = None
camera_stream_should_stop = False

_frame_buffer = []
_frame_buffer_lock = threading.Lock()
_MAX_BUFFER_SIZE = 50


def store_frame(frame):
    global _frame_buffer
    with _frame_buffer_lock:
        _frame_buffer.append(frame.copy())
        if len(_frame_buffer) > _MAX_BUFFER_SIZE:
            _frame_buffer.pop(0)


def get_buffered_frames():
    with _frame_buffer_lock:
        return list(_frame_buffer)


def clear_frame_buffer():
    global _frame_buffer
    with _frame_buffer_lock:
        _frame_buffer.clear()


def ensure_folders():
    os.makedirs("StudentDetails", exist_ok=True)
    os.makedirs(TRAINING_DIR, exist_ok=True)
    os.makedirs("TrainingImageLabel", exist_ok=True)
    os.makedirs(ATTENDANCE_DIR, exist_ok=True)


def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def legacy_date_to_iso(value):
    value = str(value).strip()
    if not value:
        return None
    for fmt in ("%d-%m-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def init_db():
    ensure_folders()
    with get_db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS students (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                name TEXT NOT NULL,
                attendance_date TEXT NOT NULL,
                attendance_time TEXT NOT NULL,
                confidence REAL,
                UNIQUE(student_id, attendance_date),
                FOREIGN KEY(student_id) REFERENCES students(id)
            )
            """
        )

        student_count = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
        if student_count == 0 and os.path.exists(STUDENT_FILE):
            df = pd.read_csv(STUDENT_FILE, dtype={"Id": str, "Name": str})
            if not df.empty:
                df = df.dropna(subset=["Id", "Name"])
                df["Id"] = df["Id"].astype(str).str.strip()
                df["Name"] = df["Name"].astype(str).map(clean_name)
                df = df[df["Id"].map(valid_student_id)]
                df["Id"] = df["Id"].map(normalize_id)
                df = df.drop_duplicates(subset=["Id"], keep="last")
                conn.executemany(
                    "INSERT OR REPLACE INTO students (id, name) VALUES (?, ?)",
                    [(row["Id"], row["Name"]) for _, row in df.iterrows()],
                )

        if os.path.exists(ATTENDANCE_DIR):
            for filename in os.listdir(ATTENDANCE_DIR):
                if not filename.startswith("Attendance_") or not filename.endswith(".xlsx"):
                    continue
                path = os.path.join(ATTENDANCE_DIR, filename)
                try:
                    df = pd.read_excel(path, dtype={"Id": str})
                except Exception:
                    continue
                if df.empty or "Id" not in df.columns or "Name" not in df.columns:
                    continue
                for _, row in df.iterrows():
                    student_id = str(row.get("Id", "")).strip()
                    if not valid_student_id(student_id):
                        continue
                    attendance_date = legacy_date_to_iso(row.get("Date", ""))
                    attendance_time = str(row.get("Time", "")).strip() or "00:00:00"
                    if not attendance_date:
                        continue
                    confidence = row.get("Confidence")
                    try:
                        confidence = float(confidence) if confidence == confidence else None
                    except Exception:
                        confidence = None
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO attendance
                        (student_id, name, attendance_date, attendance_time, confidence)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            normalize_id(student_id),
                            clean_name(row.get("Name", "")),
                            attendance_date,
                            attendance_time,
                            confidence,
                        ),
                    )


def valid_student_id(student_id):
    return bool(re.fullmatch(r"\d+", str(student_id).strip()))


def normalize_id(student_id):
    return str(int(str(student_id).strip()))


def clean_name(name):
    return re.sub(r"\s+", " ", str(name).strip())


def safe_file_name(name):
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", clean_name(name))
    return safe.strip("_") or "Student"


def parse_training_id(filename):
    stem = os.path.splitext(filename)[0]
    parts = stem.rsplit(".", 2)
    if len(parts) != 3:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def training_file_student_id(filename):
    label = parse_training_id(filename)
    if label is None:
        return None
    return str(label)


def preprocess_face(gray_face):
    face = cv2.resize(gray_face, FACE_SIZE)
    # CLAHE for better local lighting normalization (background independent)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    face = clahe.apply(face)
    return face


def crop_face_with_padding(image, x, y, w, h, padding_ratio=0.22):
    pad_w = int(w * padding_ratio)
    pad_h = int(h * padding_ratio)
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(image.shape[1], x + w + pad_w)
    y2 = min(image.shape[0], y + h + pad_h)
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


def open_camera(camera_index=0):
    backends = [
        ("CAP_DSHOW", cv2.CAP_DSHOW),
        ("DEFAULT", None),
    ]

    for name, backend in backends:
        cam = cv2.VideoCapture(camera_index, backend) if backend is not None else cv2.VideoCapture(camera_index)
        if not cam or not cam.isOpened():
            if cam:
                cam.release()
            continue

        try:
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        stable_frames = 0
        for _ in range(CAMERA_FRAME_RETRY_LIMIT):
            try:
                ret, frame = cam.read()
                if ret and frame is not None and getattr(frame, "size", 0) > 0:
                    stable_frames += 1
                    if stable_frames >= 3:
                        break
                else:
                    stable_frames = 0
            except Exception:
                stable_frames = 0
            time.sleep(0.1)

        if stable_frames >= 3:
            time.sleep(0.2)
            return cam

        cam.release()
        time.sleep(0.2)

    return None


def stop_camera_stream():
    global camera_stream_should_stop
    camera_stream_should_stop = True


def reset_camera_stream():
    global camera_stream_should_stop
    camera_stream_should_stop = False


def has_lbph():
    return hasattr(cv2, "face") and hasattr(cv2.face, "LBPHFaceRecognizer_create")


def insightface_available():
    return FaceAnalysis is not None


def ensure_color(face_image):
    if face_image is None:
        return None
    if len(face_image.shape) == 2:
        return cv2.cvtColor(face_image, cv2.COLOR_GRAY2BGR)
    return face_image


def cosine_similarity(vec_a, vec_b):
    return float(np.dot(vec_a, vec_b))


def normalize_embedding(embedding):
    embedding = np.asarray(embedding, dtype="float32").flatten()
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return None
    return embedding / norm


_arcface_onnx_session = None

def get_arcface_onnx_session():
    global _arcface_onnx_session
    if _arcface_onnx_session is not None:
        return _arcface_onnx_session
    model_path = os.path.join(ARCFACE_MODEL_ROOT, "models", ARCFACE_MODEL_NAME, "w600k_r50.onnx")
    if not os.path.exists(model_path):
        print(f"[DEBUG] ArcFace ONNX model not found at {model_path}")
        return None
    try:
        import onnxruntime as ort
        _arcface_onnx_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        print("[DEBUG] ArcFace ONNX session loaded successfully (no insightface needed)")
        return _arcface_onnx_session
    except Exception as exc:
        print(f"[DEBUG] ArcFace ONNX load failed: {exc}")
        return None


def extract_arcface_embedding_onnx(face_image):
    session = get_arcface_onnx_session()
    if session is None:
        return None
    try:
        img = cv2.resize(face_image, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = (img - 127.5) / 127.5
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: img})
        embedding = outputs[0].flatten()
        return normalize_embedding(embedding)
    except Exception as exc:
        print(f"[DEBUG] ONNX ArcFace extraction failed: {exc}")
        return None


def get_arcface_app():
    global arcface_app, arcface_init_error

    if arcface_app is not None:
        return arcface_app
    if not insightface_available():
        print("[DEBUG] ArcFace skip: insightface not installed")
        return None
    if arcface_init_error is not None:
        print(f"[DEBUG] ArcFace skip: previous init error: {arcface_init_error}")
        return None

    try:
        os.makedirs(ARCFACE_MODEL_ROOT, exist_ok=True)
        print(f"[DEBUG] Initializing ArcFace with root={os.path.abspath(ARCFACE_MODEL_ROOT)}, model={ARCFACE_MODEL_NAME}")
        arcface_app = FaceAnalysis(
            name=ARCFACE_MODEL_NAME,
            root=os.path.abspath(ARCFACE_MODEL_ROOT),
            providers=["CPUExecutionProvider"],
        )
        arcface_app.prepare(ctx_id=0, det_size=(640, 640))
        print("[DEBUG] ArcFace initialized successfully")
    except Exception as exc:
        arcface_init_error = str(exc)
        print(f"[DEBUG] ArcFace init FAILED: {exc}")
        arcface_app = None

    return arcface_app


def extract_arcface_embedding(face_image):
    color_face = ensure_color(face_image)
    if color_face is None:
        return None

    # Try insightface first
    app = get_arcface_app()
    if app is not None:
        try:
            faces = app.get(color_face)
            if faces:
                best_face = max(
                    faces,
                    key=lambda item: (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1]),
                )
                embedding = normalize_embedding(best_face.embedding)
                if embedding is not None:
                    return embedding
        except Exception:
            pass

        try:
            recognition_model = getattr(app, "models", {}).get("recognition")
            if recognition_model is not None and hasattr(recognition_model, "get_feat"):
                aligned = cv2.resize(color_face, (112, 112))
                embedding = normalize_embedding(recognition_model.get_feat(aligned))
                if embedding is not None:
                    return embedding
        except Exception:
            pass

    # Fallback to pure onnxruntime (works without insightface on Python 3.13)
    return extract_arcface_embedding_onnx(color_face)


def face_descriptor(face):
    face = cv2.resize(face, (120, 120)).astype("float32") / 255.0
    mean = float(face.mean())
    std = float(face.std()) or 1.0
    return ((face - mean) / std).reshape(-1)


class SimpleFaceRecognizer:
    def __init__(self, descriptors=None, labels=None):
        self.descriptors = np.array([] if descriptors is None else descriptors, dtype="float32")
        self.labels = np.array([] if labels is None else labels, dtype=str)

    def predict(self, face):
        if len(self.descriptors) == 0:
            return None, 999.0, 999.0

        descriptor = face_descriptor(face)
        distances = np.sqrt(((self.descriptors - descriptor) ** 2).mean(axis=1))
        nearest = np.argsort(distances)[:8]
        scores = defaultdict(list)

        for index in nearest:
            scores[str(self.labels[index])].append(float(distances[index]))

        ordered = sorted(
            ((label, sum(values) / len(values)) for label, values in scores.items()),
            key=lambda item: item[1],
        )
        best_label, best_score = ordered[0]
        second_score = ordered[1][1] if len(ordered) > 1 else 999.0
        return best_label, round(best_score * 100, 2), round(second_score * 100, 2)


class ArcFaceRecognizer:
    def __init__(self, embeddings=None, labels=None):
        self.embeddings = np.array([] if embeddings is None else embeddings, dtype="float32")
        self.labels = np.array([] if labels is None else labels, dtype=str)

    def predict(self, face):
        embedding = extract_arcface_embedding(face)
        if embedding is None or len(self.embeddings) == 0:
            return None, -1.0, -1.0

        similarities = self.embeddings @ embedding
        nearest = np.argsort(similarities)[::-1][:8]
        scores = defaultdict(list)

        for index in nearest:
            scores[str(self.labels[index])].append(float(similarities[index]))

        ordered = sorted(
            ((label, sum(values[:3]) / len(values[:3])) for label, values in scores.items()),
            key=lambda item: item[1],
            reverse=True,
        )
        best_label, best_score = ordered[0]
        second_score = ordered[1][1] if len(ordered) > 1 else -1.0
        return best_label, round(best_score, 4), round(second_score, 4)


def build_simple_recognizer(allowed_ids):
    descriptors = []
    labels = []

    if not os.path.exists(TRAINING_DIR):
        return None

    for file in os.listdir(TRAINING_DIR):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        student_id = training_file_student_id(file)
        if student_id not in allowed_ids:
            continue

        img_path = os.path.join(TRAINING_DIR, file)
        try:
            img = Image.open(img_path).convert("L")
        except Exception:
            continue

        descriptors.append(face_descriptor(preprocess_face(np.array(img, "uint8"))))
        labels.append(student_id)

    if not descriptors:
        return None

    return SimpleFaceRecognizer(descriptors, labels)


def build_arcface_recognizer(allowed_ids):
    embeddings = []
    labels = []

    # ArcFace can work via insightface OR pure onnxruntime
    arcface_ready = get_arcface_app() is not None or get_arcface_onnx_session() is not None
    if not arcface_ready:
        print("[DEBUG] build_arcface_recognizer: No ArcFace runtime available (insightface or onnxruntime)")
        return None

    total = 0
    skipped = 0
    for file in os.listdir(TRAINING_DIR):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        student_id = training_file_student_id(file)
        if student_id not in allowed_ids:
            continue

        image = cv2.imread(os.path.join(TRAINING_DIR, file))
        if image is None:
            skipped += 1
            continue

        total += 1
        embedding = extract_arcface_embedding(image)
        if embedding is None:
            skipped += 1
            continue

        embeddings.append(embedding)
        labels.append(student_id)

    print(f"[DEBUG] build_arcface_recognizer: total={total}, success={len(embeddings)}, skipped={skipped}")
    if not embeddings:
        return None

    return ArcFaceRecognizer(embeddings, labels)


def get_removed_student_ids():
    if not os.path.exists(TRAINING_DIR):
        return set()

    registered_ids = set(load_students()["Id"].astype(str))
    training_ids = set()

    for file in os.listdir(TRAINING_DIR):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        student_id = training_file_student_id(file)
        if student_id:
            training_ids.add(student_id)

    return (training_ids - registered_ids) | (set(REMOVED_STUDENT_IDS) - registered_ids)


def training_sample_counts():
    counts = Counter()
    if not os.path.exists(TRAINING_DIR):
        return {}

    for file in os.listdir(TRAINING_DIR):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        student_id = training_file_student_id(file)
        if student_id:
            counts[student_id] += 1

    return dict(counts)


def delete_student_training_images(student_id):
    removed = 0
    if not os.path.exists(TRAINING_DIR):
        return removed

    normalized_id = normalize_id(student_id)
    for file in os.listdir(TRAINING_DIR):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        if training_file_student_id(file) != normalized_id:
            continue
        try:
            os.remove(os.path.join(TRAINING_DIR, file))
            removed += 1
        except OSError:
            continue
    return removed


def clear_model_files():
    for path in (MODEL_FILE, FALLBACK_MODEL_FILE, ARCFACE_MODEL_FILE):
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                continue


def save_simple_model(faces, ids):
    descriptors = [face_descriptor(face) for face in faces]
    np.savez_compressed(FALLBACK_MODEL_FILE, descriptors=np.array(descriptors, dtype="float32"), labels=np.array(ids, dtype=str))


def save_arcface_model(recognizer):
    np.savez_compressed(
        ARCFACE_MODEL_FILE,
        embeddings=np.array(recognizer.embeddings, dtype="float32"),
        labels=np.array(recognizer.labels, dtype=str),
    )


def load_face_recognizer():
    arcface_runtime_ready = get_arcface_app() is not None or get_arcface_onnx_session() is not None
    if os.path.exists(ARCFACE_MODEL_FILE) and arcface_runtime_ready:
        data = np.load(ARCFACE_MODEL_FILE)
        recognizer = ArcFaceRecognizer(data["embeddings"], data["labels"])
        print("[DEBUG] load_face_recognizer: Loaded ArcFace model")
        return recognizer, "arcface"

    if has_lbph() and os.path.exists(MODEL_FILE):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(MODEL_FILE)
        print("[DEBUG] load_face_recognizer: Loaded LBPH model")
        return recognizer, "lbph"

    if os.path.exists(FALLBACK_MODEL_FILE):
        data = np.load(FALLBACK_MODEL_FILE)
        recognizer = SimpleFaceRecognizer(data["descriptors"], data["labels"])
        print("[DEBUG] load_face_recognizer: Loaded fallback (Simple) model")
        return recognizer, "fallback"

    print("[DEBUG] load_face_recognizer: No model found!")
    return None, None


def recognizer_student_count(recognizer, fallback_count):
    labels = getattr(recognizer, "labels", None)
    if labels is None:
        return fallback_count

    unique_labels = {str(label) for label in labels if str(label).strip()}
    return len(unique_labels) or fallback_count


def build_removed_face_recognizer():
    removed_ids = get_removed_student_ids()
    if not removed_ids:
        return None
    if insightface_available():
        recognizer = build_arcface_recognizer(removed_ids)
        if recognizer is not None:
            return recognizer, "arcface"
    recognizer = build_simple_recognizer(removed_ids)
    if recognizer is not None:
        return recognizer, "fallback"
    return None


def predict_with_backend(recognizer, backend, face_gray, face_color):
    if backend == "arcface":
        return recognizer.predict(face_color)
    if backend == "lbph":
        label, confidence = recognizer.predict(preprocess_face(face_gray))
        return str(label), float(confidence), 999.0
    return recognizer.predict(face_gray)


def arcface_threshold_for_count(registered_count):
    if registered_count <= 1:
        return 0.28
    if registered_count <= 3:
        return 0.31
    if registered_count <= 6:
        return 0.34
    return ARCFACE_SIMILARITY_THRESHOLD


def attendance_match_ok(backend, best_score, second_score, registered_count):
    if backend == "arcface":
        threshold = arcface_threshold_for_count(registered_count)
        margin = 0.02 if registered_count <= 3 else ARCFACE_SECOND_BEST_MARGIN
        if best_score < threshold:
            return False
        if second_score > -1 and (best_score - second_score) < margin:
            return False
        return True

    if backend == "lbph":
        return best_score <= CONFIDENCE_LIMIT

    return best_score <= FALLBACK_CONFIDENCE_LIMIT


def stable_match_ok(backend, vote_count, total_votes, avg_score, registered_count):
    if backend == "arcface":
        required_frames = 2 if registered_count <= 3 else MIN_CONFIDENT_FRAMES
        required_win_ratio = 0.45 if registered_count <= 3 else MIN_WIN_RATIO
    else:
        # Fallback models (LBPH/Simple) are less accurate and prone to false positives
        # Require much stronger evidence when using fallback
        required_frames = 6 if registered_count <= 3 else MIN_CONFIDENT_FRAMES + 2
        required_win_ratio = 0.65 if registered_count <= 3 else 0.70

    if vote_count < required_frames or vote_count / total_votes < required_win_ratio:
        return False

    if backend == "arcface":
        return avg_score >= arcface_threshold_for_count(registered_count)

    if backend == "lbph":
        return avg_score <= AVG_CONFIDENCE_LIMIT

    fallback_limit = 160 if registered_count == 1 else FALLBACK_AVG_CONFIDENCE_LIMIT
    return avg_score <= fallback_limit


def _process_single_frame(img):
    """Process a single frame for face recognition. Returns dict with status and match info."""
    students = load_students()
    if students.empty:
        return {"status": "error", "message": "No student is registered."}

    id_to_name = dict(zip(students["Id"].astype(str), students["Name"]))
    recognizer, backend = load_face_recognizer()
    if recognizer is None:
        return {"status": "error", "message": "Model trained nahi hai. Pehle 'Train Model' chalayein."}

    registered_count = recognizer_student_count(recognizer, len(id_to_name))
    removed_recognizer = build_removed_face_recognizer()

    # Resize for faster processing on slow CPUs (Render free tier)
    h_img, w_img = img.shape[:2]
    max_dim = 480
    if max(h_img, w_img) > max_dim:
        scale = max_dim / max(h_img, w_img)
        img_small = cv2.resize(img, (int(w_img * scale), int(h_img * scale)))
        scale_factor = 1.0 / scale
    else:
        img_small = img
        scale_factor = 1.0

    try:
        gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=FACE_DETECT_SCALE_FACTOR,
            minNeighbors=FACE_DETECT_MIN_NEIGHBORS,
            minSize=(int(FACE_DETECT_MIN_SIZE[0] * scale_factor), int(FACE_DETECT_MIN_SIZE[1] * scale_factor)),
        )
    except Exception:
        return {"status": "no_face", "message": "No face detected."}

    if len(faces) == 0:
        return {"status": "no_face", "message": "No face detected."}

    # Use original image for face crop to maintain quality
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    x_orig = int(x * scale_factor)
    y_orig = int(y * scale_factor)
    w_orig = int(w * scale_factor)
    h_orig = int(h * scale_factor)

    face_color = crop_face_with_padding(img, x_orig, y_orig, w_orig, h_orig, 0.22)
    if face_color is None:
        return {"status": "no_face", "message": "No face detected."}

    try:
        face_gray = cv2.cvtColor(face_color, cv2.COLOR_BGR2GRAY)
        # CLAHE for better local lighting normalization (background independent)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        face_gray = clahe.apply(face_gray)
        # Small blur to reduce noise
        face_gray = cv2.GaussianBlur(face_gray, (3, 3), 0)
    except Exception:
        return {"status": "no_face", "message": "No face detected."}

    if removed_recognizer is not None:
        removed_recognizer_model, removed_backend = removed_recognizer
        _, removed_score, _ = predict_with_backend(removed_recognizer_model, removed_backend, face_gray, face_color)
        removed_match = (
            removed_score >= REMOVED_FACE_ARCFACE_THRESHOLD
            if removed_backend == "arcface"
            else removed_score <= REMOVED_FACE_FALLBACK_CONFIDENCE_LIMIT
        )
        if removed_match:
            return {"status": "removed", "message": "Removed face detected.", "confidence": float(removed_score)}

    label, best_score, second_score = predict_with_backend(recognizer, backend, face_gray, face_color)
    student_id = str(label) if label is not None else None

    if student_id in id_to_name and attendance_match_ok(backend, best_score, second_score, registered_count):
        return {
            "status": "matched",
            "student_id": student_id,
            "name": id_to_name[student_id],
            "confidence": float(best_score),
            "backend": backend,
        }

    return {"status": "unmatched", "message": "Face match nahi hua."}


def load_students():
    init_db()
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT id AS Id, name AS Name FROM students ORDER BY CAST(id AS INTEGER), id"
        ).fetchall()
    if not rows:
        return pd.DataFrame(columns=["Id", "Name"])
    return pd.DataFrame([dict(row) for row in rows], columns=["Id", "Name"])


def save_students(df):
    init_db()
    df = df[["Id", "Name"]].drop_duplicates(subset=["Id"], keep="last").copy()
    df["Id"] = df["Id"].astype(str).map(normalize_id)
    df["Name"] = df["Name"].astype(str).map(clean_name)
    with get_db_connection() as conn:
        conn.execute("DELETE FROM students")
        conn.executemany(
            "INSERT INTO students (id, name) VALUES (?, ?)",
            [(row["Id"], row["Name"]) for _, row in df.iterrows()],
        )


def student_exists(student_id):
    students = load_students()
    return normalize_id(student_id) in set(students["Id"].astype(str))


def add_student(student_id, name):
    init_db()
    with get_db_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO students (id, name) VALUES (?, ?)",
            (normalize_id(student_id), clean_name(name)),
        )


def remove_student_files(student_id):
    if not os.path.exists(TRAINING_DIR):
        return 0

    removed = 0
    for file in os.listdir(TRAINING_DIR):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        if training_file_student_id(file) != student_id:
            continue
        try:
            os.remove(os.path.join(TRAINING_DIR, file))
            removed += 1
        except OSError:
            continue
    return removed


def remove_model_files():
    for path in (MODEL_FILE, FALLBACK_MODEL_FILE, ARCFACE_MODEL_FILE):
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                continue


def remove_student_from_csv(student_id):
    if not os.path.exists(STUDENT_FILE):
        return
    try:
        df = pd.read_csv(STUDENT_FILE, dtype={"Id": str, "Name": str})
        if df.empty or "Id" not in df.columns:
            return
        df = df[df["Id"].astype(str).str.strip() != student_id]
        df.to_csv(STUDENT_FILE, index=False)
    except Exception:
        pass


def delete_student(student_id):
    student_id = normalize_id(student_id)
    students = load_students()
    match = students[students["Id"].astype(str) == student_id]
    if match.empty:
        return False, "Student record not found."

    student_name = match.iloc[0]["Name"]
    removed_samples = remove_student_files(student_id)
    remove_student_from_csv(student_id)

    init_db()
    with get_db_connection() as conn:
        conn.execute("DELETE FROM attendance WHERE student_id = ?", (student_id,))
        conn.execute("DELETE FROM students WHERE id = ?", (student_id,))

    remaining_students = load_students()
    if remaining_students.empty:
        remove_model_files()
        model_message = "No students are left, so the trained face model files were cleared."
    else:
        success, train_message = train_model()
        model_message = train_message if success else f"Student removed, but model retraining needs attention: {train_message}"

    return True, f"{student_name} (ID: {student_id}) deleted. Removed {removed_samples} face samples. {model_message}"


def latest_attendance_rows(limit=200):
    init_db()
    with get_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                student_id AS Id,
                name AS Name,
                strftime('%d-%m-%Y', attendance_date) AS Date,
                attendance_time AS Time,
                ROUND(COALESCE(confidence, 0), 2) AS Confidence
            FROM attendance
            ORDER BY attendance_date DESC, attendance_time DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def total_attendance_count():
    init_db()
    with get_db_connection() as conn:
        return conn.execute("SELECT COUNT(*) FROM attendance").fetchone()[0]


def attendance_files():
    init_db()
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT DISTINCT attendance_date FROM attendance ORDER BY attendance_date"
        ).fetchall()
    return [row["attendance_date"] for row in rows]


def attendance_summary():
    students = load_students()
    total_days = len(attendance_files())
    if students.empty:
        return []

    with get_db_connection() as conn:
        present_rows = conn.execute(
            """
            SELECT student_id, COUNT(DISTINCT attendance_date) AS present_days
            FROM attendance
            GROUP BY student_id
            """
        ).fetchall()
    present_map = {row["student_id"]: row["present_days"] for row in present_rows}

    summary = []
    for _, row in students.iterrows():
        student_id = str(row["Id"])
        present_days = int(present_map.get(student_id, 0))
        summary.append(
            {
                "Id": student_id,
                "Name": row["Name"],
                "PresentDays": present_days,
                "TotalDays": total_days,
                "Percentage": round((present_days / total_days) * 100, 2) if total_days else 0,
            }
        )

    return summary


def student_attendance_stat(student_id):
    student_id = normalize_id(student_id)
    for row in attendance_summary():
        if row["Id"] == student_id:
            return row
    return {"Id": student_id, "PresentDays": 0, "TotalDays": len(attendance_files()), "Percentage": 0}


def gen_frames(max_duration=15):
    reset_camera_stream()
    clear_frame_buffer()
    cam = open_camera(0)
    if cam is None:
        return

    start_time = time.time()
    try:
        while True:
            if camera_stream_should_stop:
                break
            if time.time() - start_time > max_duration:
                break
            success, frame = cam.read()
            if not success:
                time.sleep(0.05)
                continue

            store_frame(frame)

            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
            time.sleep(0.03)
    finally:
        cam.release()


@app.route("/video_feed")
def video_feed():
    max_duration = request.args.get("max_duration", type=int, default=15)
    return Response(gen_frames(max_duration=max_duration), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stop_camera_preview", methods=["POST"])
def stop_camera_preview():
    stop_camera_stream()
    return ("", 204)


@app.route("/camera")
def camera():
    return render_template("camera.html")


@app.route("/register_camera")
def register_camera():
    if "admin" not in session:
        return redirect(url_for("login"))
    return render_template("register_camera.html")


def train_model():
    ensure_folders()
    if not os.path.exists(TRAINING_DIR) or len(os.listdir(TRAINING_DIR)) == 0:
        return False, "No training images found."

    registered_ids = set(load_students()["Id"].astype(str))
    if not registered_ids:
        return False, "No registered students found."

    counts = training_sample_counts()
    faces = []
    ids = []

    for file in os.listdir(TRAINING_DIR):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        label = parse_training_id(file)
        if label is None:
            continue
        if str(label) not in registered_ids:
            continue

        img_path = os.path.join(TRAINING_DIR, file)
        try:
            img = Image.open(img_path).convert("L")
        except Exception:
            continue

        faces.append(preprocess_face(np.array(img, "uint8")))
        ids.append(label)

    if not faces:
        return False, "No matching training images were found for the registered students. Please add the student again using **Register Student**."

    save_simple_model(faces, ids)
    trained_counts = Counter(str(student_id) for student_id in ids)
    summary = ", ".join(f"ID {student_id}: {count}" for student_id, count in sorted(trained_counts.items()))

    arcface_ready = get_arcface_app() is not None or get_arcface_onnx_session() is not None
    if arcface_ready:
        arcface_recognizer = build_arcface_recognizer(registered_ids)
        if arcface_recognizer is not None and len(arcface_recognizer.labels) > 0:
            save_arcface_model(arcface_recognizer)
            arcface_counts = Counter(str(student_id) for student_id in arcface_recognizer.labels)
            arcface_summary = ", ".join(
                f"ID {student_id}: {count}" for student_id, count in sorted(arcface_counts.items())
            )
            return True, f"ArcFace model trained successfully. {arcface_summary}"

    if has_lbph():
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, np.array(ids))
        recognizer.save(MODEL_FILE)
        return True, f"Model trained with {len(faces)} face samples. {summary}"

    return True, f"Model trained with {len(faces)} face samples. {summary}. OpenCV fallback model use ho raha hai."


def active_model_file():
    arcface_runtime_ready = get_arcface_app() is not None or get_arcface_onnx_session() is not None
    if arcface_runtime_ready and os.path.exists(ARCFACE_MODEL_FILE):
        return ARCFACE_MODEL_FILE
    if has_lbph() and os.path.exists(MODEL_FILE):
        return MODEL_FILE
    if os.path.exists(FALLBACK_MODEL_FILE):
        return FALLBACK_MODEL_FILE
    if insightface_available():
        return ARCFACE_MODEL_FILE
    if has_lbph():
        return MODEL_FILE
    return FALLBACK_MODEL_FILE


def latest_registered_training_mtime(registered_ids):
    if not os.path.exists(TRAINING_DIR):
        return 0

    latest_mtime = 0
    for file in os.listdir(TRAINING_DIR):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        student_id = training_file_student_id(file)
        if student_id not in registered_ids:
            continue
        try:
            latest_mtime = max(latest_mtime, os.path.getmtime(os.path.join(TRAINING_DIR, file)))
        except OSError:
            continue
    return latest_mtime


def model_needs_training():
    students = load_students()
    registered_ids = set(students["Id"].astype(str))
    if not registered_ids:
        return False

    recognizer, _ = load_face_recognizer()
    active_model = active_model_file()
    if recognizer is None or not os.path.exists(active_model):
        return True

    sample_counts = training_sample_counts()
    if any(sample_counts.get(student_id, 0) == 0 for student_id in registered_ids):
        return True

    if os.path.getmtime(active_model) < latest_registered_training_mtime(registered_ids):
        return True

    return False


@app.route("/train")
def train():
    if "admin" not in session:
        return redirect(url_for("login"))

    success, message = train_model()
    return render_template("result.html", success=success, title="Training", message=message, back_url=url_for("home"))


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        if request.form["username"] == "Rishu" and request.form["password"] == "Rishu@123":
            session["admin"] = True
            return redirect(url_for("home"))
        error = "Invalid username or password."

    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.pop("admin", None)
    return redirect(url_for("login"))


@app.route("/")
def home():
    students = load_students()
    attendance_rows = latest_attendance_rows(limit=10)
    attendance_total_count = total_attendance_count()
    attendance_stats = attendance_summary()
    model_ready = not model_needs_training()
    recognizer, backend = load_face_recognizer()
    backend_name = backend if backend else "none"
    return render_template(
        "index.html",
        students=students.to_dict("records"),
        attendance_rows=attendance_rows,
        attendance_total_count=attendance_total_count,
        attendance_stats=attendance_stats,
        model_ready=model_ready,
        student_count=len(students),
        backend=backend_name,
    )


@app.route("/app_status")
def app_status():
    students = load_students()
    arcface_runtime_ready = get_arcface_app() is not None if insightface_available() else False
    return {
        "version": APP_VERSION,
        "students": students.to_dict("records"),
        "model_needs_training": model_needs_training(),
        "removed_student_ids": sorted(get_removed_student_ids()),
        "face_backend": "arcface" if insightface_available() else ("opencv_lbph" if has_lbph() else "numpy_fallback"),
        "training_sample_counts": training_sample_counts(),
        "arcface_ready": os.path.exists(ARCFACE_MODEL_FILE),
        "arcface_runtime_ready": arcface_runtime_ready,
        "arcface_error": arcface_init_error,
    }


@app.route("/result")
def result_page():
    success = request.args.get("success", "0") == "1"
    title = request.args.get("title", "Result")
    message = request.args.get("message", "")
    return render_template("result.html", success=success, title=title, message=message, back_url=url_for("home"))


@app.route("/process_frame", methods=["POST"])
def process_frame():
    if "image" not in request.files:
        return jsonify({"status": "error", "message": "No image received"})

    file = request.files["image"]
    img_bytes = file.read()
    if not img_bytes:
        return jsonify({"status": "error", "message": "Empty image received"})

    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"status": "error", "message": "Invalid image format"})

    result = _process_single_frame(img)
    return jsonify(result)


@app.route("/mark_attendance_ajax", methods=["POST"])
def mark_attendance_ajax():
    data = request.get_json(force=True, silent=True) or {}
    student_id = str(data.get("student_id", "")).strip()
    name = str(data.get("name", "")).strip()
    confidence = data.get("confidence", 0)

    if not student_id or not name:
        return jsonify({"success": False, "message": "Invalid student data."})

    if not student_exists(student_id):
        return jsonify({"success": False, "message": "Student registered nahi hai."})

    now = datetime.now()
    attendance_date = now.strftime("%Y-%m-%d")
    attendance_time = now.strftime("%H:%M:%S")
    display_date = now.strftime("%d-%m-%Y")

    with get_db_connection() as conn:
        existing = conn.execute(
            "SELECT 1 FROM attendance WHERE student_id = ? AND attendance_date = ?",
            (student_id, attendance_date),
        ).fetchone()
        already_marked = existing is not None
        if not already_marked:
            conn.execute(
                """
                INSERT INTO attendance
                (student_id, name, attendance_date, attendance_time, confidence)
                VALUES (?, ?, ?, ?, ?)
                """,
                (student_id, name, attendance_date, attendance_time, round(float(confidence), 2)),
            )
            try:
                ensure_folders()
                excel_filename = f"Attendance_{display_date}.xlsx"
                excel_path = os.path.join(ATTENDANCE_DIR, excel_filename)
                new_row = pd.DataFrame([{
                    "Id": student_id,
                    "Name": name,
                    "Date": display_date,
                    "Time": attendance_time,
                    "Confidence": round(float(confidence), 2),
                }])
                if os.path.exists(excel_path):
                    existing_df = pd.read_excel(excel_path, dtype={"Id": str})
                    combined = pd.concat([existing_df, new_row], ignore_index=True)
                    combined = combined.drop_duplicates(subset=["Id", "Date"], keep="last")
                    combined.to_excel(excel_path, index=False)
                else:
                    new_row.to_excel(excel_path, index=False)
            except Exception as exc:
                print(f"[DEBUG] Excel save failed: {exc}")

    return jsonify({"success": True, "message": f"Attendance marked for {name} (ID: {student_id}) at {attendance_time}."})


@app.route("/capture_register_frame", methods=["POST"])
def capture_register_frame():
    if "admin" not in session:
        return jsonify({"success": False, "message": "Admin login required."})

    student_id = session.get("temp_id")
    name = session.get("temp_name")
    if not student_id or not name:
        return jsonify({"success": False, "message": "Registration session expired. Please start again."})

    if "image" not in request.files:
        return jsonify({"success": False, "message": "No image received"})

    file = request.files["image"]
    img_bytes = file.read()
    if not img_bytes:
        return jsonify({"success": False, "message": "Empty image received"})

    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"success": False, "message": "Invalid image format"})

    # Resize for faster processing on slow CPUs
    h_img, w_img = img.shape[:2]
    max_dim = 480
    if max(h_img, w_img) > max_dim:
        scale = max_dim / max(h_img, w_img)
        img_small = cv2.resize(img, (int(w_img * scale), int(h_img * scale)))
        scale_factor = 1.0 / scale
    else:
        img_small = img
        scale_factor = 1.0

    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=5,
        minSize=(int(70 * scale_factor), int(70 * scale_factor)),
    )

    if len(faces) == 0:
        return jsonify({"success": False, "message": "No face detected in frame"})

    x, y, w, h = faces[0]
    x_orig = int(x * scale_factor)
    y_orig = int(y * scale_factor)
    w_orig = int(w * scale_factor)
    h_orig = int(h * scale_factor)
    padded_face = crop_face_with_padding(img, x_orig, y_orig, w_orig, h_orig, 0.22)
    if padded_face is None:
        return jsonify({"success": False, "message": "Face crop failed"})

    face = cv2.resize(padded_face, (224, 224))
    ensure_folders()

    # Find next sample number for this student
    existing_samples = [
        f for f in os.listdir(TRAINING_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
        and training_file_student_id(f) == str(student_id)
    ]
    sample_num = len(existing_samples) + 1

    filename = f"{safe_file_name(name)}.{student_id}.{sample_num}.jpg"
    filepath = os.path.join(TRAINING_DIR, filename)
    cv2.imwrite(filepath, face)

    return jsonify({"success": True, "sample": sample_num})


@app.route("/process_register_ajax", methods=["POST"])
def process_register_ajax():
    if "admin" not in session:
        return jsonify({"success": False, "message": "Admin login required."})

    student_id = session.get("temp_id")
    name = session.get("temp_name")

    if not student_id or not name:
        return jsonify({"success": False, "message": "Registration session expired."})

    if student_exists(student_id):
        session.pop("temp_id", None)
        session.pop("temp_name", None)
        return jsonify({"success": False, "message": f"ID {student_id} already registered."})

    # Count saved samples
    ensure_folders()
    sample_count = sum(
        1 for f in os.listdir(TRAINING_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
        and training_file_student_id(f) == str(student_id)
    )

    if sample_count < 12:
        return jsonify({"success": False, "message": f"Only {sample_count} samples captured. Need at least 12. Please try again with better lighting."})

    add_student(student_id, name)
    success, train_message = train_model()
    session.pop("temp_id", None)
    session.pop("temp_name", None)

    return jsonify({
        "success": success,
        "message": f"{name} (ID: {student_id}) registered successfully. {train_message}"
    })


@app.route("/process_attendance")
def process_attendance():
    if model_needs_training():
        return render_template(
            "result.html",
            success=False,
            title="Attendance",
            message="The model needs to be updated. Please log in as admin first and run **Train Model**.",
            back_url=url_for("home"),
        )

    students = load_students()
    if students.empty:
        return render_template(
            "result.html",
            success=False,
            title="Attendance",
            message="No student is registered.",
            back_url=url_for("home"),
        )

    id_to_name = dict(zip(students["Id"].astype(str), students["Name"]))
    recognizer, backend = load_face_recognizer()
    if recognizer is None:
        return render_template(
            "result.html",
            success=False,
            title="Attendance",
            message="The model could not be loaded. Please log in as admin first and run **Train Model**.",
            back_url=url_for("home"),
        )

    removed_recognizer = build_removed_face_recognizer()
    registered_count = recognizer_student_count(recognizer, len(id_to_name))

    def _process_attendance_frame(img):
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=FACE_DETECT_SCALE_FACTOR,
                minNeighbors=FACE_DETECT_MIN_NEIGHBORS,
                minSize=FACE_DETECT_MIN_SIZE,
            )
        except Exception:
            return "no_face", None, None

        if len(faces) == 0:
            return "no_face", None, None

        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        face_color = crop_face_with_padding(img, x, y, w, h, 0.22)
        if face_color is None:
            return "no_face", None, None

        try:
            face_gray = cv2.cvtColor(face_color, cv2.COLOR_BGR2GRAY)
            face_gray = cv2.equalizeHist(face_gray)
        except Exception:
            return "no_face", None, None

        if removed_recognizer is not None:
            removed_recognizer_model, removed_backend = removed_recognizer
            _, removed_score, _ = predict_with_backend(removed_recognizer_model, removed_backend, face_gray, face_color)
            removed_match = (
                removed_score >= REMOVED_FACE_ARCFACE_THRESHOLD
                if removed_backend == "arcface"
                else removed_score <= REMOVED_FACE_FALLBACK_CONFIDENCE_LIMIT
            )
            if removed_match:
                return "removed", None, removed_score

        label, best_score, second_score = predict_with_backend(recognizer, backend, face_gray, face_color)
        student_id = str(label) if label is not None else None

        if student_id in id_to_name and attendance_match_ok(backend, best_score, second_score, registered_count):
            return "matched", student_id, best_score

        return "unmatched", None, None

    cam = None
    votes = Counter()
    confidences = defaultdict(list)
    removed_votes = 0
    frames_checked = 0
    face_detected_frames = 0

    # Try buffered frames from video feed preview first
    buffered_frames = get_buffered_frames()
    use_buffered = len(buffered_frames) >= 8

    if use_buffered:
        for img in buffered_frames:
            status, student_id, best_score = _process_attendance_frame(img)

            if status == "removed":
                removed_votes += 1
                if removed_votes >= REMOVED_FACE_MIN_FRAMES:
                    break
                continue

            if status in ("matched", "unmatched"):
                face_detected_frames += 1

            if status == "matched":
                votes[student_id] += 1
                confidences[student_id].append(best_score)
                current_votes = votes[student_id]
                current_avg = sum(confidences[student_id]) / len(confidences[student_id])
                if stable_match_ok(backend, current_votes, sum(votes.values()), current_avg, registered_count) and current_votes >= FAST_MATCH_TARGET:
                    break

            if sum(votes.values()) >= FAST_MATCH_TARGET + 1:
                break

        frames_checked = len(buffered_frames)
    else:
        for camera_attempt in range(3):
            stop_camera_stream()
            time.sleep(1.0 + camera_attempt * 0.6)
            cam = open_camera(0)
            if cam is None:
                continue

            stabilized = 0
            for _ in range(15):
                try:
                    ret, _ = cam.read()
                    if ret:
                        stabilized += 1
                except Exception:
                    pass
                time.sleep(0.07)

            if stabilized < 5:
                cam.release()
                time.sleep(0.3)
                continue

            votes = Counter()
            confidences = defaultdict(list)
            removed_votes = 0
            frames_checked = 0
            face_detected_frames = 0

            for _ in range(150):
                try:
                    ret, img = cam.read()
                except Exception:
                    time.sleep(0.05)
                    continue
                if not ret or img is None or getattr(img, "size", 0) == 0:
                    time.sleep(0.05)
                    continue

                frames_checked += 1
                status, student_id, best_score = _process_attendance_frame(img)

                if status == "removed":
                    removed_votes += 1
                    if removed_votes >= REMOVED_FACE_MIN_FRAMES:
                        break
                    continue

                if status in ("matched", "unmatched"):
                    face_detected_frames += 1

                if status == "matched":
                    votes[student_id] += 1
                    confidences[student_id].append(best_score)
                    current_votes = votes[student_id]
                    current_avg = sum(confidences[student_id]) / len(confidences[student_id])
                    if stable_match_ok(backend, current_votes, sum(votes.values()), current_avg, registered_count) and current_votes >= FAST_MATCH_TARGET:
                        break

                if sum(votes.values()) >= FAST_MATCH_TARGET + 1:
                    break

            cam.release()

            if removed_votes >= REMOVED_FACE_MIN_FRAMES:
                break
            if frames_checked >= 8:
                break

    print(f"[DEBUG] Backend={backend}, buffered={len(buffered_frames)}, use_buffered={use_buffered}, frames_checked={frames_checked}, face_detected={face_detected_frames}, votes={dict(votes)}, removed_votes={removed_votes}")

    if not use_buffered and cam is None:
        return render_template(
            "result.html",
            success=False,
            title="Attendance",
            message="Camera open nahi ho pa raha. Dusri app me camera use ho raha ho to usse band karke phir try karein.",
            back_url=url_for("home"),
        )

    if removed_votes >= REMOVED_FACE_MIN_FRAMES:
        return render_template(
            "result.html",
            success=False,
            title="Attendance",
            message="This face matches a removed student ID. Attendance has not been marked.",
            back_url=url_for("home"),
        )

    if frames_checked < 8:
        return render_template(
            "result.html",
            success=False,
            title="Attendance",
            message="Camera frames stable nahi mil rahe. Camera preview ya dusri app band karke phir try karein.",
            back_url=url_for("home"),
        )

    if face_detected_frames == 0:
        return render_template(
            "result.html",
            success=False,
            title="Attendance",
            message="Face detect nahi ho paaya. Camera ke saamne seedha aayein aur light improve karein.",
            back_url=url_for("home"),
        )

    if not votes:
        return render_template(
            "result.html",
            success=False,
            title="Attendance",
            message="The face was not recognized. Please look straight at the camera, ensure proper lighting, and try again.",
            back_url=url_for("home"),
        )

    student_id, vote_count = votes.most_common(1)[0]
    total_votes = sum(votes.values())
    avg_conf = sum(confidences[student_id]) / len(confidences[student_id])

    if not stable_match_ok(backend, vote_count, total_votes, avg_conf, registered_count):
        return render_template(
            "result.html",
            success=False,
            title="Attendance",
            message="A stable face match was not found. Please come a bit closer and try again in proper lighting.",
            back_url=url_for("home"),
        )

    name = id_to_name[student_id]
    attendance_date = datetime.now().strftime("%Y-%m-%d")
    attendance_time = datetime.now().strftime("%H:%M:%S")
    display_date = datetime.now().strftime("%d-%m-%Y")

    init_db()
    with get_db_connection() as conn:
        existing = conn.execute(
            "SELECT 1 FROM attendance WHERE student_id = ? AND attendance_date = ?",
            (student_id, attendance_date),
        ).fetchone()
        already_marked = existing is not None
        if not already_marked:
            conn.execute(
                """
                INSERT INTO attendance
                (student_id, name, attendance_date, attendance_time, confidence)
                VALUES (?, ?, ?, ?, ?)
                """,
                (student_id, name, attendance_date, attendance_time, round(avg_conf, 2)),
            )
            # Save to daily Excel file
            try:
                ensure_folders()
                excel_filename = f"Attendance_{display_date}.xlsx"
                excel_path = os.path.join(ATTENDANCE_DIR, excel_filename)
                new_row = pd.DataFrame([{
                    "Id": student_id,
                    "Name": name,
                    "Date": display_date,
                    "Time": attendance_time,
                    "Confidence": round(avg_conf, 2),
                }])
                if os.path.exists(excel_path):
                    existing_df = pd.read_excel(excel_path, dtype={"Id": str})
                    combined = pd.concat([existing_df, new_row], ignore_index=True)
                    combined = combined.drop_duplicates(subset=["Id", "Date"], keep="last")
                    combined.to_excel(excel_path, index=False)
                else:
                    new_row.to_excel(excel_path, index=False)
            except Exception as exc:
                print(f"[DEBUG] Excel save failed: {exc}")

    stat = student_attendance_stat(student_id)
    return render_template(
        "result.html",
        success=True,
        title="Attendance",
        message=(
            f"{name} (ID: {student_id}), your today's attendance is already marked."
            if already_marked
            else f"{name} (ID: {student_id}) attendance marked."
        ),
        back_url=url_for("home"),
        stats=[
            {"label": "Present Days", "value": stat["PresentDays"]},
            {"label": "Total Days", "value": stat["TotalDays"]},
            {"label": "Attendance", "value": f"{stat['Percentage']}%"},
        ],
    )


@app.route("/register", methods=["POST"])
def register():
    if "admin" not in session:
        return redirect(url_for("login"))

    student_id = request.form["id"].strip()
    name = clean_name(request.form["name"])

    if not valid_student_id(student_id):
        return render_template(
            "result.html",
            success=False,
            title="Register Student",
            message="The Student ID should contain only numbers.",
            back_url=url_for("home"),
        )

    student_id = normalize_id(student_id)
    if student_exists(student_id):
        return render_template(
            "result.html",
            success=False,
            title="Register Student",
            message=f"ID {student_id} already registered .Each student must have a unique ID.",
            back_url=url_for("home"),
        )

    session["temp_id"] = student_id
    session["temp_name"] = name
    return redirect(url_for("register_camera"))


@app.route("/process_register")
def process_register():
    if "admin" not in session:
        return redirect(url_for("login"))

    student_id = session.get("temp_id")
    name = session.get("temp_name")

    if not student_id or not name:
        return redirect(url_for("home"))

    if student_exists(student_id):
        return render_template(
            "result.html",
            success=False,
            title="Register Student",
            message=f"ID {student_id} Already registered. The registration has been canceled.",
            back_url=url_for("home"),
        )

    cam = None
    for camera_attempt in range(3):
        stop_camera_stream()
        time.sleep(0.5 + camera_attempt * 0.4)
        cam = open_camera(0)
        if cam is not None:
            break

    if cam is None:
        session.pop("temp_id", None)
        session.pop("temp_name", None)
        return render_template(
            "result.html",
            success=False,
            title="Register Student",
            message="Camera open nahi ho pa raha. Camera ko use karne wali dusri app band karke phir try karein.",
            back_url=url_for("home"),
        )
    sample = 0
    ensure_folders()

    for _ in range(650):
        ret, img = cam.read()
        if not ret:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5, minSize=(70, 70))

        for (x, y, w, h) in faces[:1]:
            sample += 1
            padded_face = crop_face_with_padding(img, x, y, w, h, 0.22)
            if padded_face is None:
                continue
            face = cv2.resize(padded_face, (224, 224))
            cv2.imwrite(os.path.join(TRAINING_DIR, f"{safe_file_name(name)}.{student_id}.{sample}.jpg"), face)

        if sample >= SAMPLE_COUNT:
            break

    cam.release()
    cv2.destroyAllWindows()

    if sample < 12:
        return render_template(
            "result.html",
            success=False,
            title="Register Student",
            message="Not enough face samples were captured (need 12+). Please check the lighting and camera angle, then try again.",
            back_url=url_for("home"),
        )

    add_student(student_id, name)
    success, train_message = train_model()
    session.pop("temp_id", None)
    session.pop("temp_name", None)

    return render_template(
        "result.html",
        success=success,
        title="Register Student",
        message=f"{name} (ID: {student_id}) add ho gaya. {train_message}",
        back_url=url_for("home"),
    )


@app.route("/delete_student", methods=["POST"])
def delete_student_route():
    if "admin" not in session:
        return redirect(url_for("login"))

    student_id = request.form.get("student_id", "").strip()
    if not valid_student_id(student_id):
        return render_template(
            "result.html",
            success=False,
            title="Delete Student",
            message="The selected student ID is invalid.",
            back_url=url_for("home"),
        )

    success, message = delete_student(student_id)
    if session.get("temp_id") == normalize_id(student_id):
        session.pop("temp_id", None)
        session.pop("temp_name", None)

    return render_template(
        "result.html",
        success=success,
        title="Delete Student",
        message=message,
        back_url=url_for("home"),
    )


@app.route("/retrain")
def retrain_model():
    if "admin" not in session:
        return redirect(url_for("login"))
    students = load_students()
    if students.empty:
        return render_template(
            "result.html",
            success=False,
            title="Retrain Model",
            message="Koyi student registered nahi hai. Pehle student register karein.",
            back_url=url_for("home"),
        )
    success, train_message = train_model()
    return render_template(
        "result.html",
        success=success,
        title="Retrain Model",
        message=train_message,
        back_url=url_for("home"),
    )


@app.route("/sync_attendance")
def sync_attendance_route():
    if "admin" not in session:
        return redirect(url_for("login"))
    init_db()
    count = 0
    if os.path.exists(ATTENDANCE_DIR):
        for filename in os.listdir(ATTENDANCE_DIR):
            if not filename.startswith("Attendance_") or not filename.endswith(".xlsx"):
                continue
            path = os.path.join(ATTENDANCE_DIR, filename)
            try:
                df = pd.read_excel(path, dtype={"Id": str})
            except Exception:
                continue
            if df.empty or "Id" not in df.columns or "Name" not in df.columns:
                continue
            with get_db_connection() as conn:
                for _, row in df.iterrows():
                    student_id = str(row.get("Id", "")).strip()
                    if not valid_student_id(student_id):
                        continue
                    attendance_date = legacy_date_to_iso(row.get("Date", ""))
                    attendance_time = str(row.get("Time", "")).strip() or "00:00:00"
                    if not attendance_date:
                        continue
                    confidence = row.get("Confidence")
                    try:
                        confidence = float(confidence) if confidence == confidence else None
                    except Exception:
                        confidence = None
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO attendance
                        (student_id, name, attendance_date, attendance_time, confidence)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            normalize_id(student_id),
                            clean_name(row.get("Name", "")),
                            attendance_date,
                            attendance_time,
                            confidence,
                        ),
                    )
                    count += 1
    return render_template(
        "result.html",
        success=True,
        title="Sync Attendance",
        message=f"Attendance sync complete. {count} records imported from Excel files.",
        back_url=url_for("home"),
    )


@app.route("/db_view")
def db_view():
    if "admin" not in session:
        return redirect(url_for("login"))
    init_db()
    with get_db_connection() as conn:
        students_rows = conn.execute("SELECT * FROM students ORDER BY CAST(id AS INTEGER), id").fetchall()
        attendance_rows = conn.execute(
            """
            SELECT id, student_id, name,
                   strftime('%d-%m-%Y', attendance_date) AS date,
                   attendance_time AS time,
                   ROUND(COALESCE(confidence, 0), 2) AS confidence
            FROM attendance
            ORDER BY attendance_date DESC, attendance_time DESC, id DESC
            LIMIT 300
            """
        ).fetchall()
        total_attendance = conn.execute("SELECT COUNT(*) FROM attendance").fetchone()[0]
        total_students = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
        distinct_dates = conn.execute("SELECT COUNT(DISTINCT attendance_date) FROM attendance").fetchone()[0]
    return render_template(
        "db_view.html",
        students=[dict(row) for row in students_rows],
        attendance=[dict(row) for row in attendance_rows],
        total_attendance=total_attendance,
        total_students=total_students,
        distinct_dates=distinct_dates,
    )


# Initialize on module import so gunicorn workers have folders and DB ready
try:
    ensure_folders()
    init_db()
except Exception as exc:
    print(f"[WARNING] Startup initialization issue: {exc}")


@app.route("/health")
def health():
    return {"status": "ok", "version": APP_VERSION}


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False)
