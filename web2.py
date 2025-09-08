import os
import cv2
import time
import smtplib
import argparse
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
from dotenv import load_dotenv
from ultralytics import YOLO

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

# --- Constants ---
WINDOW_NAME = "YOLOv8 PPE Detection"
ALERT_THRESHOLD_SECONDS = 10.0
EMAIL_COOLDOWN_SECONDS = 120.0
EMAIL_SENT_MSG_DURATION_SECONDS = 5.0

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# --- Class Color Mapping (BGR) ---
CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "Hardhat": (255, 0, 0),
    "Mask": (0, 255, 0),
    "NO-Hardhat": (0, 0, 255),
    "NO-Mask": (255, 255, 0),
    "NO-Safety Vest": (255, 0, 255),
    "Person": (0, 255, 255),
    "Safety Cone": (128, 0, 128),
    "Safety Vest": (128, 128, 0),
    "Machinery": (0, 128, 128),
    "Vehicle": (128, 128, 128)
}
DEFAULT_COLOR = (100, 100, 100)


def setup_config() -> argparse.Namespace:
    """Loads configuration from .env file and command-line arguments."""
    load_dotenv()
    parser = argparse.ArgumentParser(description="Real-time PPE Detection with Email Alerts.")
    # ... (rest of the function is unchanged)
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.getenv("MODEL_PATH", "Model/ppe.pt"),
        help="Path to the YOLOv8 model file."
    )
    parser.add_argument(
        "--source",
        type=str,
        default=os.getenv("VIDEO_SOURCE", "0"),
        help="Video source (webcam index or video file path)."
    )
    parser.add_argument("--sender-email", type=str, default=os.getenv("SENDER_EMAIL"))
    parser.add_argument("--receiver-email", type=str, default=os.getenv("RECEIVER_EMAIL"))
    parser.add_argument("--email-password", type=str, default=os.getenv("EMAIL_PASSWORD"))
    parser.add_argument("--smtp-server", type=str, default=os.getenv("SMTP_SERVER", "smtp.gmail.com"))
    parser.add_argument("--smtp-port", type=int, default=int(os.getenv("SMTP_PORT", 587)))
    args = parser.parse_args()
    if args.source.isdigit():
        args.source = int(args.source)
    return args


def draw_text_with_background(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_scale: float = 0.5,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    padding: int = 5
) -> None:
    """Draws text with a semi-transparent background on a frame."""
    # ... (this function is unchanged)
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    bg_x1, bg_y1 = x - padding, y - text_h - padding
    bg_x2, bg_y2 = x + text_w + padding, y + padding
    overlay = frame.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

# --- MODIFICATION: Function now accepts subject and body ---
def send_email_alert(
    config: argparse.Namespace,
    image_path: Path,
    alert_subject: str,
    alert_body: str
) -> None:
    """Constructs and sends an email with a dynamic message and an image attachment."""
    if not all([config.sender_email, config.receiver_email, config.email_password]):
        logging.warning("Email credentials not fully configured. Skipping email alert.")
        return

    message = MIMEMultipart()
    message["From"] = config.sender_email
    message["To"] = config.receiver_email
    message["Subject"] = alert_subject # Use dynamic subject
    
    message.attach(MIMEText(alert_body, "plain")) # Use dynamic body
    
    try:
        with open(image_path, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={image_path.name}")
            message.attach(part)
        
        with smtplib.SMTP(config.smtp_server, config.smtp_port) as server:
            server.starttls()
            server.login(config.sender_email, config.email_password)
            server.sendmail(config.sender_email, config.receiver_email, message.as_string())
        logging.info(f"Email alert sent successfully to {config.receiver_email}")
    except Exception as e:
        logging.error(f"An unexpected error occurred while sending email: {e}")


def run_inference(config: argparse.Namespace):
    """Main loop for running video capture, inference, and displaying results."""
    # ... (model and video capture setup is unchanged)
    try:
        model = YOLO(config.model_path)
    except Exception as e:
        logging.error(f"Failed to load YOLO model: {e}")
        return
    cap = cv2.VideoCapture(config.source)
    if not cap.isOpened():
        logging.error(f"Error: Unable to access video source '{config.source}'.")
        return
    logging.info(f"Video source '{config.source}' opened. Press 'q' to exit.")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # --- MODIFICATION: State variables updated ---
    alert_trigger_start_time: Optional[float] = None
    last_email_sent_time: float = 0
    email_sent_confirmation_time: float = 0
    last_detected_violations: List[str] = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("End of video stream or cannot read frame.")
                break

            person_detected = False
            # --- MODIFICATION: Track all violations in a set for the current frame ---
            current_violations = set()
            detection_counts = {name: 0 for name in CLASS_COLORS.keys()}

            results = model(frame)

            for result in results:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    class_name = model.names.get(cls_id, "Unknown")
                    
                    if class_name == "Person":
                        person_detected = True

                    # --- MODIFICATION: Add any "NO-" class to the violations set ---
                    if class_name.startswith("NO-"):
                        # Clean up the name for the alert message
                        missing_item = class_name.replace("NO-", "")
                        current_violations.add(missing_item)

                    # Drawing logic (unchanged)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detection_counts[class_name] = detection_counts.get(class_name, 0) + 1
                    color = CLASS_COLORS.get(class_name, DEFAULT_COLOR)
                    label = f"{class_name} {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    draw_text_with_background(frame, label, (x1, y1 - 10), bg_color=color)

            # --- MODIFICATION: Updated Alert Logic ---
            is_alert_condition = person_detected and len(current_violations) > 0
            
            if is_alert_condition:
                # Convert set to sorted list for consistent ordering
                detected_violations_list = sorted(list(current_violations))

                if alert_trigger_start_time is None or last_detected_violations != detected_violations_list:
                    alert_trigger_start_time = time.time() # Start/reset timer on new violation set
                    last_detected_violations = detected_violations_list

                elapsed_time = time.time() - alert_trigger_start_time
                
                # Draw alert status on screen
                violations_str = ", ".join(last_detected_violations)
                alert_text = f"ALERT: Missing {violations_str} ({int(elapsed_time)}s)"
                draw_text_with_background(
                    frame, alert_text, (10, frame.shape[0] - 20), 
                    font_scale=0.6, text_color=(255, 255, 255), bg_color=(0, 0, 200)
                )

                # Check if we should send an email
                if elapsed_time >= ALERT_THRESHOLD_SECONDS:
                    if (time.time() - last_email_sent_time) >= EMAIL_COOLDOWN_SECONDS:
                        image_path = Path("alert_frame.jpg")
                        cv2.imwrite(str(image_path), frame)
                        
                        # --- MODIFICATION: Build dynamic email subject and body ---
                        if len(last_detected_violations) > 1:
                            alert_subject = "Alert: Multiple PPE Violations Detected"
                            items_text = f"The following items were not detected: {violations_str}."
                        else:
                            alert_subject = f"Alert: {violations_str} Not Detected"
                            items_text = f"A {violations_str} was not detected."

                        alert_body = (
                            f"A person was detected without required PPE for a sustained period.\n\n"
                            f"{items_text}\n\n"
                            "Please review the attached image for details."
                        )

                        # Send email in a separate thread
                        email_thread = threading.Thread(
                            target=send_email_alert,
                            args=(config, image_path, alert_subject, alert_body)
                        )
                        email_thread.start()
                        
                        last_email_sent_time = time.time()
                        email_sent_confirmation_time = time.time()
            else:
                alert_trigger_start_time = None # Reset timer
                last_detected_violations = []

            # On-screen Information (unchanged)
            y_pos = 30
            for name, count in detection_counts.items():
                if count > 0:
                    text = f"{name}s: {count}"
                    draw_text_with_background(frame, text, (10, y_pos))
                    y_pos += 30

            if time.time() - email_sent_confirmation_time < EMAIL_SENT_MSG_DURATION_SECONDS:
                draw_text_with_background(
                    frame, "Email Sent!", (frame.shape[1] - 150, 30), 
                    text_color=(0, 255, 0), bg_color=(0, 0, 0)
                )
            
            cv2.imshow(WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logging.info("Resources released and windows closed.")


def main():
    """Main entry point of the script."""
    config = setup_config()
    run_inference(config)


if __name__ == "__main__":
    main()