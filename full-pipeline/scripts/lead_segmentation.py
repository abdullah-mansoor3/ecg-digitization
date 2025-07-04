import os
import cv2
from ultralytics import YOLO
import yaml

# Load config from YAML
with open('./configs/lead_segmentation.yaml', 'r') as f:
    config = yaml.safe_load(f)

CONF_THRESHOLD = config['conf_threshold']
MODEL_PATH = config['model_path']
LEFT_LABELS = config['left_labels']
RIGHT_LABELS = config['right_labels']


def init_model(model_path=MODEL_PATH):
    """
    Initialize the YOLO model for lead segmentation.
    Args:
        model_path: Path to the YOLO model weights.
    Returns:
        model: Initialized YOLO model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    
    model = YOLO(model_path)
    return model


def sort_wave_boxes(boxes, row_threshold=30):
    """
    Sorts boxes in row-major order (top-to-bottom, left-to-right), robust to overlaps.
    Each box: [x1, y1, x2, y2]
    row_threshold: max vertical distance between centers to consider boxes in the same row.
    """
    # Compute vertical centers
    boxes_with_centers = []
    for box in boxes:
        x1, y1, x2, y2 = box
        y_center = (y1 + y2) / 2
        x_center = (x1 + x2) / 2
        boxes_with_centers.append((box, y_center, x_center))

    # Sort by y_center
    boxes_with_centers.sort(key=lambda b: b[1])

    # Group into rows
    rows = []
    for box, y_center, x_center in boxes_with_centers:
        placed = False
        for row in rows:
            # If y_center is close to the first box in the row, add to this row
            if abs(row[0][1] - y_center) < row_threshold:
                row.append((box, y_center, x_center))
                placed = True
                break
        if not placed:
            rows.append([(box, y_center, x_center)])

    # Sort each row by x_center
    for row in rows:
        row.sort(key=lambda b: b[2])

    # Flatten rows
    sorted_boxes = [b[0] for row in rows for b in row]
    return sorted_boxes

def inference_and_label_and_crop(model, input_image_path, output_dir, conf_threshold=CONF_THRESHOLD):
    """
    Perform inference on a single image, label detected boxes, and save cropped leads.
    Args:
        model: YOLO model
        input_image_path: Path to input image
        output_dir: Directory to save cropped leads
        conf_threshold: Confidence threshold for detection
    Returns:
        cropped_leads: List of (cropped lead image, label)
        labeled_boxes_paths: List of saved image paths
    """
    os.makedirs(output_dir, exist_ok=True)
    results = model(input_image_path)[0]
    img = cv2.imread(input_image_path)
    base_name = os.path.splitext(os.path.basename(input_image_path))[0]

    wave_boxes = []
    for box in results.boxes:
        cls_id = int(box.cls[0]) # 0 Class ID for wave segments
        conf = float(box.conf[0])
        if conf >= conf_threshold and cls_id == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            wave_boxes.append([x1, y1, x2, y2])

    wave_boxes = sort_wave_boxes(wave_boxes)

    cropped_leads = []
    cropped_leads_paths = []
    for idx, box in enumerate(wave_boxes):
        label = LEFT_LABELS[idx] if idx < len(LEFT_LABELS) else RIGHT_LABELS[idx - len(LEFT_LABELS)]

        # Crop and save
        x1, y1, x2, y2 = box
        crop = img[y1:y2, x1:x2]
        save_path = os.path.join(output_dir, f"{base_name}_{label}.jpg")
        cv2.imwrite(save_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 100])

        cropped_leads_paths.append(save_path)
        cropped_leads.append((crop, label))

    return cropped_leads, cropped_leads_paths