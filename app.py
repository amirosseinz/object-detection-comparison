# -*- coding: utf-8 -*-
import os
import csv
import time
import uuid
import logging
import json
import sys
import platform
from datetime import datetime
from functools import wraps
from flask import Flask, request, render_template, redirect, url_for, jsonify, flash, send_from_directory, g
from flask_cors import CORS
from werkzeug.utils import secure_filename
import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io

# --- Request ID Generator for Tracing ---
def generate_request_id():
    """Generate a unique request ID for tracing."""
    return str(uuid.uuid4())[:8]

# --- Custom Logging Formatter with Request ID ---
class RequestIdFilter(logging.Filter):
    def filter(self, record):
        record.request_id = getattr(g, 'request_id', 'N/A') if has_request_context() else 'STARTUP'
        return True

def has_request_context():
    """Check if we're in a request context."""
    try:
        from flask import has_request_context as flask_has_request_context
        return flask_has_request_context()
    except:
        return False

# --- Basic Logging Setup ---
# Configure root logger with a simple format (for werkzeug and other libraries)
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Create app-specific logger with request ID support
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handler with request_id format for our app logger only
app_handler = logging.StreamHandler()
app_handler.setLevel(logging.INFO)
app_handler.setFormatter(logging.Formatter('%(asctime)s - [%(request_id)s] - %(levelname)s - %(message)s'))
app_handler.addFilter(RequestIdFilter())

# Remove default handlers and add our custom one
logger.handlers = []
logger.addHandler(app_handler)
logger.propagate = False  # Don't pass to root logger

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Secret key management with environment variable support
app.secret_key = os.environ.get('SECRET_KEY', os.environ.get('FLASK_SECRET_KEY')) or os.urandom(24)
if not os.environ.get('SECRET_KEY') and not os.environ.get('FLASK_SECRET_KEY'):
    logger.warning("SECRET_KEY not set in environment. Using random key (sessions won't persist across restarts).") 

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'originals')
app.config['STATIC_FOLDER'] = os.path.join(BASE_DIR, 'static')
app.config['CLASSIFIED_FOLDER_A'] = os.path.join(app.config['STATIC_FOLDER'], 'classified', 'model_a')
app.config['CLASSIFIED_FOLDER_B'] = os.path.join(app.config['STATIC_FOLDER'], 'classified', 'model_b')
app.config['RESULTS_DETAILS_FOLDER'] = os.path.join(BASE_DIR, 'results_details')
app.config['RESULTS_CSV'] = os.path.join(BASE_DIR, 'detection_results.csv')
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# TensorFlow Serving URLs
TF_SERVING_URL_MODEL_A = os.environ.get('TF_SERVING_URL_MODEL_A', 'http://ssd-serving:8501/v1/models/ssd:predict')
TF_SERVING_URL_MODEL_B = os.environ.get('TF_SERVING_URL_MODEL_B', 'http://faster-rcnn-serving:8501/v1/models/faster_rcnn:predict')

# Label map for both models
LABEL_MAP = {
    1: 'StruthioCamelus',
    2: 'PhacochoerusAfricanus',
    3: 'Pantheraleo'
}

# Confidence threshold for displaying detections
CONFIDENCE_THRESHOLD = 0.5

# --- Cross-Platform Font Loading ---
def get_font(size=15):
    """
    Get a font with cross-platform fallback support.
    Tries multiple common font paths before falling back to default.
    """
    font_candidates = []
    
    system = platform.system().lower()
    
    if system == 'windows':
        font_candidates = [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/tahoma.ttf",
            "C:/Windows/Fonts/calibri.ttf",
            "arial.ttf"
        ]
    elif system == 'darwin':  # macOS
        font_candidates = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/SFNSText.ttf",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "arial.ttf"
        ]
    else:  # Linux and others
        font_candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
            "arial.ttf",
            "DejaVuSans.ttf"
        ]
    
    for font_path in font_candidates:
        try:
            return ImageFont.truetype(font_path, size)
        except (IOError, OSError):
            continue
    
    # Fall back to default font
    logger.warning("Could not load any TrueType font, using default bitmap font.")
    return ImageFont.load_default()

# --- TensorFlow Serving Health Check ---
def check_tf_serving_health():
    """Check if TensorFlow Serving endpoints are reachable."""
    health_status = {'model_a': False, 'model_b': False}
    
    # Extract base URLs (remove /v1/models/... part for health check)
    model_a_base = TF_SERVING_URL_MODEL_A.rsplit('/v1/models/', 1)[0]
    model_b_base = TF_SERVING_URL_MODEL_B.rsplit('/v1/models/', 1)[0]
    
    for name, base_url in [('model_a', model_a_base), ('model_b', model_b_base)]:
        try:
            response = requests.get(f"{base_url}/v1/models/{'ssd' if name == 'model_a' else 'faster_rcnn'}", timeout=5)
            if response.status_code == 200:
                health_status[name] = True
                logger.info(f"TF Serving {name} is healthy")
            else:
                logger.warning(f"TF Serving {name} returned status {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"TF Serving {name} is unreachable: {e}")
    
    return health_status

# --- Setup: Create directories and CSV ---
def setup_application():
    """Ensures necessary directories and CSV file exist and are writable."""
    directories = [
        app.config['UPLOAD_FOLDER'],
        app.config['CLASSIFIED_FOLDER_A'],
        app.config['CLASSIFIED_FOLDER_B'],
        app.config['RESULTS_DETAILS_FOLDER']
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            # Verify directory is writable
            test_file = os.path.join(directory, '.write_test')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            logger.info(f"Directory verified writable: {directory}")
        except OSError as e:
            logger.critical(f"Directory {directory} is not writable: {e}")
            raise RuntimeError(f"Required directory not writable: {directory}")
    
    init_csv()
    
    # Perform TF Serving health check on startup (non-blocking)
    logger.info("Checking TensorFlow Serving availability...")
    health = check_tf_serving_health()
    if not health['model_a']:
        logger.warning(f"Model A (SSD) TF Serving is not reachable at {TF_SERVING_URL_MODEL_A}")
    if not health['model_b']:
        logger.warning(f"Model B (Faster R-CNN) TF Serving is not reachable at {TF_SERVING_URL_MODEL_B}")
    if not any(health.values()):
        logger.warning("Neither TF Serving endpoint is reachable. Image processing will fail until services are available.")

def init_csv():
    """Initializes the CSV file if it doesn't exist."""
    csv_path = app.config['RESULTS_CSV']
    if not os.path.exists(csv_path):
        logger.info(f"Creating new CSV file at {csv_path}")
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    'image_id', 'timestamp', 'original_filename', 'original_filepath', 'file_size',
                    'model_a_filepath', 'model_a_inference_time', 'model_a_detection_count',
                    'model_b_filepath', 'model_b_inference_time', 'model_b_detection_count'
                ])
        except IOError as e:
            logger.error(f"Failed to create or write initial CSV header to {csv_path}: {e}", exc_info=True)
            raise

try:
    setup_application()
except Exception as e:
    logger.critical(f"Application setup failed: {e}", exc_info=True)

# --- Request ID Middleware ---
@app.before_request
def before_request():
    """Assign a unique request ID to each request for tracing."""
    g.request_id = generate_request_id()
    g.start_time = time.time()

# --- Helper Functions ---
def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def save_results(image_id, original_filename, original_filepath, file_size, model_a_data, model_b_data):
    """Saves the processing results summary to the CSV file."""
    csv_path = app.config['RESULTS_CSV']
    try:
        with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                image_id,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
                original_filename,
                original_filepath,
                file_size,
                model_a_data.get('output_filepath', ''),
                model_a_data.get('inference_time', 0.0),
                model_a_data.get('detection_count', 0),
                model_b_data.get('output_filepath', ''),
                model_b_data.get('inference_time', 0.0),
                model_b_data.get('detection_count', 0)
            ])
        logging.info(f"Successfully saved results summary for image_id: {image_id} to {csv_path}")
    except IOError as e:
        logging.error(f"Failed to append results for image_id {image_id} to CSV {csv_path}: {e}", exc_info=True)
    except Exception as e:
        logging.error(f"Unexpected error saving results summary for image_id {image_id}: {e}", exc_info=True)

def get_static_url(filepath):
    """Converts a file system path within the static folder to a web-accessible static URL."""
    if not filepath or not isinstance(filepath, str) or not os.path.exists(filepath):
        return None
    try:
        abs_filepath = os.path.abspath(filepath)
        abs_static_folder = os.path.abspath(app.config['STATIC_FOLDER'])
        if not abs_filepath.startswith(abs_static_folder):
            logging.warning(f"Filepath {filepath} is not within the static folder {abs_static_folder}.")
            return None
        relative_path = os.path.relpath(abs_filepath, abs_static_folder)
        try:
            return url_for('static', filename=relative_path.replace(os.sep, '/'))
        except RuntimeError:
            logging.warning("get_static_url called outside request context, constructing URL manually.")
            return '/static/' + relative_path.replace(os.sep, '/')
    except Exception as e:
        logging.error(f"Error generating static url for {filepath}: {e}", exc_info=True)
        return None

def process_image_with_model(image_filepath, model_url, output_folder, image_id, model_suffix):
    """
    Processes an image with a specific model, saves the processed image (if detections),
    saves detection details to a JSON file, and returns results.
    """
    start_time = time.time()
    output_filepath = None
    detection_count = 0
    detections_list = []
    error_message = None
    success = False
    details_saved = False
    details_filepath = os.path.join(app.config['RESULTS_DETAILS_FOLDER'], f"{image_id}_{model_suffix}.json")

    try:
        if not os.path.exists(image_filepath):
            raise FileNotFoundError(f"Input image not found at {image_filepath}")

        img = Image.open(image_filepath).convert('RGB')
        img_array = np.array(img)
        payload = {"instances": [img_array.tolist()]}

        logging.info(f"[{image_id}-{model_suffix}] Sending request to model at {model_url}")
        response = requests.post(model_url, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        logging.info(f"[{image_id}-{model_suffix}] Received response from model at {model_url}")

        predictions = result.get('predictions', [])
        if not predictions or not isinstance(predictions, list) or not predictions[0]:
            logging.warning(f"[{image_id}-{model_suffix}] Model returned no or invalid predictions.")
            boxes, scores, classes = [], [], []
        else:
            pred_data = predictions[0]
            if isinstance(pred_data, dict):
                boxes = pred_data.get('detection_boxes', [])
                scores = pred_data.get('detection_scores', [])
                classes = pred_data.get('detection_classes', [])
            else:
                logging.error(f"[{image_id}-{model_suffix}] Unrecognized prediction structure: {pred_data}")
                boxes, scores, classes = [], [], []

        valid_detections = [
            (box, score, cls_id) for box, score, cls_id in zip(boxes, scores, classes)
            if isinstance(score, (float, int)) and score > CONFIDENCE_THRESHOLD and isinstance(box, list) and len(box) == 4
        ]
        detection_count = len(valid_detections)
        logging.info(f"[{image_id}-{model_suffix}] Found {detection_count} detections above threshold")

        for box, score, cls_id_float in valid_detections:
            ymin, xmin, ymax, xmax = box
            class_id = int(cls_id_float)
            class_name = LABEL_MAP.get(class_id, f"Unknown_Class_{class_id}")
            confidence = float(score)
            detections_list.append({
                'class_id': class_id,
                'class_name': class_name,
                'confidence': round(confidence, 4),
                'bbox_normalized': [float(ymin), float(xmin), float(ymax), float(xmax)]
            })

        try:
            os.makedirs(os.path.dirname(details_filepath), exist_ok=True)
            with open(details_filepath, 'w', encoding='utf-8') as f_json:
                json.dump(detections_list, f_json, indent=2)
            details_saved = True
            logging.info(f"[{image_id}-{model_suffix}] Saved detection details to: {details_filepath}")
        except IOError as e:
            logging.error(f"[{image_id}-{model_suffix}] Failed to save detection details: {e}", exc_info=True)
            error_message = "Failed to save detection details."

        if detection_count > 0:
            img_draw = img.copy()
            draw = ImageDraw.Draw(img_draw)
            font = get_font(15)  # Use cross-platform font loading

            img_width, img_height = img.size
            for det_data in detections_list:
                ymin, xmin, ymax, xmax = det_data['bbox_normalized']
                xmin_px, xmax_px = int(xmin * img_width), int(xmax * img_width)
                ymin_px, ymax_px = int(ymin * img_height), int(ymax * img_height)
                class_name = det_data['class_name']
                confidence = det_data['confidence']
                draw.rectangle([(xmin_px, ymin_px), (xmax_px, ymax_px)], outline="red", width=3)
                text = f"{class_name}: {confidence:.2f}"
                if hasattr(draw, 'textbbox'):
                    text_bbox = draw.textbbox((0, 0), text, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                else:
                    text_width, text_height = 10 * len(text), 10
                text_bg_coords = [(xmin_px, ymin_px - text_height - 4), (xmin_px + text_width + 4, ymin_px)]
                if text_bg_coords[0][1] < 0:
                    text_bg_coords = [(xmin_px, ymax_px), (xmin_px + text_width + 4, ymax_px + text_height + 4)]
                draw.rectangle(text_bg_coords, fill="red")
                draw.text((text_bg_coords[0][0] + 2, text_bg_coords[0][1] + 2), text, fill="white", font=font)

            original_basename = os.path.basename(image_filepath)
            name, ext = os.path.splitext(original_basename)
            output_filename = f"{image_id}_{model_suffix}_processed{ext}"
            output_filepath = os.path.join(output_folder, output_filename)
            os.makedirs(output_folder, exist_ok=True)
            img_draw.save(output_filepath)
            logging.info(f"[{image_id}-{model_suffix}] Saved processed image to: {output_filepath}")
        else:
            logging.info(f"[{image_id}-{model_suffix}] No detections, no processed image saved.")

        success = True

    except Exception as e:
        error_message = f"Error processing image: {e}"
        logging.error(f"[{image_id}-{model_suffix}] {error_message}", exc_info=True)

    inference_time = time.time() - start_time
    logging.info(f"[{image_id}-{model_suffix}] Processing took {inference_time:.4f} seconds.")

    final_success = success and error_message is None

    return {
        'success': final_success,
        'error': error_message,
        'output_filepath': output_filepath,
        'inference_time': inference_time,
        'detection_count': detection_count,
        'detections': detections_list,
        'details_saved': details_saved
    }

def get_metrics():
    """Calculates and returns overall metrics from the CSV file."""
    csv_path = app.config['RESULTS_CSV']
    default_metrics = {
        'total_images': 0,
        'model_a': {'avg_time': 0.0, 'total_detections': 0, 'avg_detections': 0.0},
        'model_b': {'avg_time': 0.0, 'total_detections': 0, 'avg_detections': 0.0}
    }
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        logging.warning(f"Metrics CSV not found or empty at {csv_path}.")
        return default_metrics

    total_a_time, total_b_time = 0.0, 0.0
    total_a_detections, total_b_detections = 0, 0
    valid_rows_count = 0

    try:
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            required_fields = ['model_a_inference_time', 'model_b_inference_time',
                               'model_a_detection_count', 'model_b_detection_count']
            if not all(field in reader.fieldnames for field in required_fields):
                logging.error(f"Metrics CSV missing required columns: {reader.fieldnames}")
                return default_metrics

            for row in reader:
                try:
                    a_time = float(row.get('model_a_inference_time', '0') or '0')
                    b_time = float(row.get('model_b_inference_time', '0') or '0')
                    a_det = int(row.get('model_a_detection_count', '0') or '0')
                    b_det = int(row.get('model_b_detection_count', '0') or '0')
                    total_a_time += a_time
                    total_b_time += b_time
                    total_a_detections += a_det
                    total_b_detections += b_det
                    valid_rows_count += 1
                except ValueError as e:
                    logging.warning(f"Invalid numeric data in row: {row}, error: {e}")
    except Exception as e:
        logging.error(f"Error calculating metrics: {e}", exc_info=True)
        return default_metrics

    if valid_rows_count == 0:
        return default_metrics

    return {
        'total_images': valid_rows_count,
        'model_a': {
            'avg_time': total_a_time / valid_rows_count,
            'total_detections': total_a_detections,
            'avg_detections': float(total_a_detections) / valid_rows_count
        },
        'model_b': {
            'avg_time': total_b_time / valid_rows_count,
            'total_detections': total_b_detections,
            'avg_detections': float(total_b_detections) / valid_rows_count
        }
    }

def get_image_result_data(image_id=None):
    """Gets data for a specific image ID or recent images from the CSV."""
    csv_path = app.config['RESULTS_CSV']
    details_folder = app.config['RESULTS_DETAILS_FOLDER']

    if not os.path.exists(csv_path):
        return None if image_id else []

    results, found_specific = [], None
    try:
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            if not reader.fieldnames:
                return None if image_id else []

            for row in reader:
                row_data = {k: row.get(k, '') for k in reader.fieldnames}
                row_data['file_size'] = int(row.get('file_size') or 0)
                row_data['model_a_inference_time'] = float(row.get('model_a_inference_time') or 0.0)
                row_data['model_a_detection_count'] = int(row.get('model_a_detection_count') or 0)
                row_data['model_b_inference_time'] = float(row.get('model_b_inference_time') or 0.0)
                row_data['model_b_detection_count'] = int(row.get('model_b_detection_count') or 0)
                row_data['model_a_detections'] = []
                row_data['model_b_detections'] = []
                row_data['original_unique_filename'] = os.path.basename(row_data.get('original_filepath', ''))
                row_data['model_a_processed_filename'] = os.path.basename(row_data.get('model_a_filepath', ''))
                row_data['model_b_processed_filename'] = os.path.basename(row_data.get('model_b_filepath', ''))

                current_image_id = row_data.get('image_id')
                if image_id and current_image_id == image_id:
                    details_path_a = os.path.join(details_folder, f"{image_id}_model_a.json")
                    if os.path.exists(details_path_a):
                        with open(details_path_a, 'r', encoding='utf-8') as f_json_a:
                            row_data['model_a_detections'] = json.load(f_json_a)
                    details_path_b = os.path.join(details_folder, f"{image_id}_model_b.json")
                    if os.path.exists(details_path_b):
                        with open(details_path_b, 'r', encoding='utf-8') as f_json_b:
                            row_data['model_b_detections'] = json.load(f_json_b)
                    found_specific = row_data

                results.append(row_data)

    except Exception as e:
        logging.error(f"Error processing CSV {csv_path}: {e}", exc_info=True)
        return None if image_id else []

    def generate_urls(item_data):
        try:
            item_data['original_url'] = url_for('serve_original_image', filename=item_data['original_unique_filename']) if item_data.get('original_unique_filename') else None
            model_a_path = item_data.get('model_a_filepath') or (os.path.join(app.config['CLASSIFIED_FOLDER_A'], item_data['model_a_processed_filename']) if item_data.get('model_a_processed_filename') else None)
            model_b_path = item_data.get('model_b_filepath') or (os.path.join(app.config['CLASSIFIED_FOLDER_B'], item_data['model_b_processed_filename']) if item_data.get('model_b_processed_filename') else None)
            item_data['model_a_static_url'] = get_static_url(model_a_path)
            item_data['model_b_static_url'] = get_static_url(model_b_path)
        except Exception as e:
            logging.error(f"Error generating URLs: {e}", exc_info=True)
        return item_data

    if image_id:
        return generate_urls(found_specific) if found_specific else None

    results.sort(key=lambda x: datetime.strptime(x.get('timestamp', '1970-01-01'), '%Y-%m-%d %H:%M:%S.%f'), reverse=True)
    recent_results = results[:10]
    return [generate_urls(item) for item in recent_results]

# --- Routes ---
@app.route('/')
def index():
    """Renders the main upload page."""
    return render_template('index.html')

@app.route('/originals/<filename>')
def serve_original_image(filename):
    """Serves original images from the UPLOAD_FOLDER."""
    if '..' in filename or filename.startswith('/') or filename.startswith('\\'):
        logging.warning(f"Attempted directory traversal: {filename}")
        return "Forbidden", 403
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.isfile(file_path):
            return "File not found", 404
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=False)
    except Exception as e:
        logging.error(f"Error serving file {filename}: {e}", exc_info=True)
        return "Internal Server Error", 500

@app.route('/inference', methods=['POST', 'GET'])
def inference():
    """Handles image upload, processing, saving results, and redirecting."""
    if request.method == 'GET' and 'sample' in request.args:
        sample_filename = request.args.get('sample')
        sample_path = os.path.join(app.config['STATIC_FOLDER'], 'originals', sample_filename)
        if os.path.exists(sample_path):
            image_id = str(uuid.uuid4())
            original_filename = secure_filename(sample_filename)
            file_extension = original_filename.rsplit('.', 1)[1].lower() if '.' in original_filename else 'jpg'
            destination_filename = f"{image_id}.{file_extension}"
            destination_path = os.path.join(app.config['UPLOAD_FOLDER'], destination_filename)
            import shutil
            shutil.copy2(sample_path, destination_path)
            model_a_results = process_image_with_model(destination_path, TF_SERVING_URL_MODEL_A, app.config['CLASSIFIED_FOLDER_A'], image_id, "model_a")
            model_b_results = process_image_with_model(destination_path, TF_SERVING_URL_MODEL_B, app.config['CLASSIFIED_FOLDER_B'], image_id, "model_b")
            file_size = os.path.getsize(destination_path)
            save_results(image_id, original_filename, destination_path, file_size, model_a_results, model_b_results)
            return redirect(url_for('results', id=image_id))

    image_id = str(uuid.uuid4())
    if 'file' not in request.files or request.files['file'].filename == '':
        flash('No file selected.', 'error')
        return redirect(url_for('index'))

    file = request.files['file']
    if file and allowed_file(file.filename):
        original_filename = secure_filename(file.filename)
        unique_filename = f"{image_id}_{original_filename}"
        original_filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(original_filepath)
        file_size = os.path.getsize(original_filepath)
        model_a_results = process_image_with_model(original_filepath, TF_SERVING_URL_MODEL_A, app.config['CLASSIFIED_FOLDER_A'], image_id, "model_a")
        model_b_results = process_image_with_model(original_filepath, TF_SERVING_URL_MODEL_B, app.config['CLASSIFIED_FOLDER_B'], image_id, "model_b")
        save_results(image_id, original_filename, original_filepath, file_size, model_a_results, model_b_results)
        a_ok = model_a_results.get('success', False)
        b_ok = model_b_results.get('success', False)
        a_dets = model_a_results.get('detection_count', 0)
        b_dets = model_b_results.get('detection_count', 0)
        if a_ok and b_ok:
            flash(f"Image processed. Model A found {a_dets} objects, Model B found {b_dets}.", 'success')
        else:
            errors = []
            if not a_ok: errors.append(f"Model A failed: {model_a_results.get('error', 'Unknown error')}")
            if not b_ok: errors.append(f"Model B failed: {model_b_results.get('error', 'Unknown error')}")
            flash(f"Errors: {'; '.join(errors)}.", 'warning')
        return redirect(url_for('results', id=image_id))

    flash('File type not allowed.', 'error')
    return redirect(url_for('index'))

@app.route('/results')
def results():
    """Renders the results page."""
    image_id = request.args.get('id')
    if image_id:
        image_data = get_image_result_data(image_id=image_id)
        if image_data:
            return render_template('results.html', image_data=image_data, recent_images=[], show_specific=True, show_recent=False)
        flash(f"No image found with ID: {image_id}. Showing recent results.", 'warning')
    recent_images = get_image_result_data(image_id=None)
    if not recent_images:
        flash("No recent image results found yet.", 'info')
    return render_template('results.html', image_data=None, recent_images=recent_images, show_specific=False, show_recent=True)

@app.route('/metrics')
def metrics():
    """Renders the metrics page showing overall model performance."""
    metrics_data = get_metrics()
    if not metrics_data:
        metrics_data = {
            'total_images': 0,
            'model_a': {'avg_time': 0.0, 'total_detections': 0, 'avg_detections': 0.0},
            'model_b': {'avg_time': 0.0, 'total_detections': 0, 'avg_detections': 0.0}
        }
        flash("Could not calculate metrics.", "warning")
    return render_template('metrics.html', metrics=metrics_data)

# --- API Endpoints ---
@app.route('/api/metrics', methods=['GET'])
def api_metrics():
    """API endpoint to get overall metrics in JSON format."""
    metrics_data = get_metrics()
    if not metrics_data:
        return jsonify({'error': 'Could not calculate metrics'}), 500
    return jsonify(metrics_data)

@app.route('/api/comparison/<image_id>', methods=['GET'])
def api_comparison(image_id):
    """API endpoint to get data for a specific image ID in JSON format."""
    image_data = get_image_result_data(image_id=image_id)
    if not image_data:
        return jsonify({'error': f'Image with ID {image_id} not found'}), 404
    response_data = {
        'image': {
            'image_id': image_data.get('image_id'),
            'filename': image_data.get('original_filename'),
            'original_url': image_data.get('original_url'),
            'upload_time': image_data.get('timestamp'),
            'file_size': image_data.get('file_size')
        },
        'inferences': {
            'model_a': {
                'inference_time': image_data.get('model_a_inference_time'),
                'detection_count': image_data.get('model_a_detection_count'),
                'result_static_url': image_data.get('model_a_static_url'),
                'detections': image_data.get('model_a_detections', [])
            },
            'model_b': {
                'inference_time': image_data.get('model_b_inference_time'),
                'detection_count': image_data.get('model_b_detection_count'),
                'result_static_url': image_data.get('model_b_static_url'),
                'detections': image_data.get('model_b_detections', [])
            }
        }
    }
    return jsonify(response_data)

# --- Error Handlers ---
@app.errorhandler(404)
def page_not_found(e):
    if request.path.startswith('/api/'):
        return jsonify(error="Resource not found"), 404
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    logger.error(f"500 Error: {e}", exc_info=True)
    if request.path.startswith('/api/'):
        return jsonify(error="Internal server error", message=str(e)), 500
    return render_template('500.html', error=str(e)), 500

@app.errorhandler(405)
def method_not_allowed(e):
    if request.path.startswith('/api/'):
        return jsonify(error="Method not allowed"), 405
    flash("Method not allowed.", "error")
    return redirect(url_for('index'))

# --- Health Check Endpoint for Docker ---
@app.route('/health')
def health_check():
    """Health check endpoint for container orchestration."""
    tf_health = check_tf_serving_health()
    status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'tf_serving': {
            'model_a_available': tf_health['model_a'],
            'model_b_available': tf_health['model_b']
        }
    }
    # Return 200 even if TF Serving is down (app itself is healthy)
    return jsonify(status), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)