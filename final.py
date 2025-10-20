import os
import cv2
import time
import smtplib
import argparse
import logging
import threading
import json
import sqlite3
from datetime import datetime
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
ALERT_THRESHOLD_SECONDS = 5.0
EMAIL_COOLDOWN_SECONDS = 120.0
COMPLIANCE_COOLDOWN_SECONDS = 10.0
EMAIL_SENT_MSG_DURATION_SECONDS = 5.0
EMPLOYEE_DB_PATH = "employees.json"
DATABASE_PATH = "ppe_events.db"
ALERT_IMAGE_DIR = "alert_images"

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# --- ANSI Color Codes for Terminal ---
class TermColors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'

# --- Banner Functions ---
def print_big_red_banner(reason_lines: List[str]):
    try:
        width = os.get_terminal_size().columns
    except OSError:
        width = 80
    banner_width = int(width * 0.75)
    if banner_width < 50: banner_width = 50
    print(f"{TermColors.BOLD}{TermColors.RED}")
    print("*" * banner_width)
    print(f"*{"ATTENDANCE HAS NOT BEEN MARKED!".center(banner_width - 2)}*")
    print("*" + ("-" * (banner_width - 2)) + "*")
    for line in reason_lines:
        print(f"*{line.center(banner_width - 2)}*")
    print("*" * banner_width)
    print(f"{TermColors.ENDC}")

def print_big_green_banner(reason_lines: List[str]):
    try:
        width = os.get_terminal_size().columns
    except OSError:
        width = 80
    banner_width = int(width * 0.75)
    if banner_width < 50: banner_width = 50
    print(f"{TermColors.BOLD}{TermColors.GREEN}")
    print("*" * banner_width)
    print(f"*{"ATTENDANCE HAS BEEN MARKED!".center(banner_width - 2)}*")
    print("*" + ("-" * (banner_width - 2)) + "*")
    for line in reason_lines:
        print(f"*{line.center(banner_width - 2)}*")
    print("*" * banner_width)
    print(f"{TermColors.ENDC}")

def print_big_yellow_prompt():
    try:
        width = os.get_terminal_size().columns
    except OSError:
        width = 80
    banner_width = int(width * 0.75)
    if banner_width < 50: banner_width = 50
    print(f"\n{TermColors.BOLD}{TermColors.YELLOW}")
    print("*" * banner_width)
    print(f"*{"Press '1' and ENTER to resume monitoring...".center(banner_width - 2)}*")
    print("*" * banner_width)
    print(f"{TermColors.ENDC}")

def print_big_input_prompt(prompt_lines: List[str]) -> str:
    try:
        width = os.get_terminal_size().columns
    except OSError:
        width = 80
    banner_width = int(width * 0.75)
    if banner_width < 50: banner_width = 50
    print(f"\n{TermColors.BOLD}{TermColors.YELLOW}")
    print("*" * banner_width)
    for line in prompt_lines:
        print(f"*{line.center(banner_width - 2)}*")
    print("*" * banner_width)
    print(f"{TermColors.ENDC}", end='')
    return input(f"{TermColors.BOLD}{TermColors.YELLOW} > {TermColors.ENDC}").strip()

# --- Database Functions ---
# <-- MODIFICATION: Updated setup_database to add the new column if it doesn't exist
def setup_database(db_path: str = DATABASE_PATH) -> Optional[sqlite3.Connection]:
    """Initializes the SQLite database and creates/updates the events table."""
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 1. Create table if it doesn't exist (with the new column)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                employee_id TEXT NOT NULL,
                employee_name TEXT,
                event_type TEXT NOT NULL CHECK(event_type IN ('COMPLIANCE', 'VIOLATION')),
                attendance_marked TEXT NOT NULL CHECK(attendance_marked IN ('YES', 'NO')),
                details TEXT,
                image_path TEXT
            )
        """)
        
        # 2. Check if the column exists (for old databases) and add it if not
        cursor.execute("PRAGMA table_info(events)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'attendance_marked' not in columns:
            logging.info("Updating database schema: Adding 'attendance_marked' column...")
            cursor.execute("""
                ALTER TABLE events 
                ADD COLUMN attendance_marked TEXT 
                NOT NULL 
                DEFAULT 'NO' 
                CHECK(attendance_marked IN ('YES', 'NO'))
            """)
            
            # 3. Update existing data for consistency
            # Old compliance records should be marked as 'YES'
            cursor.execute("UPDATE events SET attendance_marked = 'YES' WHERE event_type = 'COMPLIANCE'")
            logging.info("Database schema updated and old records patched.")

        conn.commit()
        logging.info(f"Database initialized successfully at {db_path}")
        return conn
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
        if conn:
            conn.close()
        return None
# <-- END MODIFICATION

# <-- MODIFICATION: Added 'attendance_marked' parameter
def log_event_to_db(conn: Optional[sqlite3.Connection], employee_id: str, employee_name: str, event_type: str, attendance_marked: str, details: Optional[str] = None, image_path: Optional[Path] = None):
    """Logs an event (compliance or violation) to the database."""
    if not conn:
        logging.warning("Database connection is not available. Skipping log.")
        return
    try:
        cursor = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            INSERT INTO events (timestamp, employee_id, employee_name, event_type, attendance_marked, details, image_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (timestamp, employee_id, employee_name, event_type, attendance_marked, details, str(image_path) if image_path else None))
        # <-- MODIFICATION: Added 'attendance_marked' to INSERT
        conn.commit()
        logging.info(f"Successfully logged {event_type} for employee {employee_id} to database.")
    except sqlite3.Error as e:
        logging.error(f"Failed to log event to database: {e}")
# <-- END MODIFICATION

# --- Class Definitions ---
EQUIPPABLE_ITEMS = {"Hardhat", "Mask", "Safety Vest"}
CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "Hardhat": (0, 255, 0), "Mask": (0, 255, 0), "Safety Vest": (0, 255, 0),
    "NO-Hardhat": (0, 0, 255), "NO-Mask": (0, 0, 255), "NO-Safety Vest": (0, 0, 255),
    "Person": (0, 255, 255), "Safety Cone": (128, 0, 128), "Machinery": (0, 128, 128),
    "Vehicle": (128, 128, 128)
}
DEFAULT_COLOR = (100, 100, 100)

def setup_config() -> argparse.Namespace:
    """Loads configuration from .env file and command-line arguments."""
    load_dotenv()
    parser = argparse.ArgumentParser(description="Real-time PPE Detection with Email Alerts.")
    parser.add_argument("--model-path", type=str, default=os.getenv("MODEL_PATH", "Model/ppe.pt"), help="Path to the YOLOv8 model file.")
    parser.add_argument("--source", type=str, default=os.getenv("VIDEO_SOURCE", "0"), help="Video source (webcam index or video file path).")
    parser.add_argument("--sender-email", type=str, default=os.getenv("SENDER_EMAIL"))
    parser.add_argument("--receiver-email", type=str, default=os.getenv("RECEIVER_EMAIL"), help="Fallback/supervisor email address.")
    parser.add_argument("--email-password", type=str, default=os.getenv("EMAIL_PASSWORD"))
    parser.add_argument("--smtp-server", type=str, default=os.getenv("SMTP_SERVER", "smtp.gmail.com"))
    parser.add_argument("--smtp-port", type=int, default=int(os.getenv("SMTP_PORT", 587)))
    args = parser.parse_args()
    if args.source.isdigit():
        args.source = int(args.source)
    return args

def load_employee_db() -> Dict[str, Dict[str, str]]:
    """Loads the employee ID-to-email mapping from a JSON file."""
    try:
        with open(EMPLOYEE_DB_PATH, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning(f"{EMPLOYEE_DB_PATH} not found.")
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from {EMPLOYEE_DB_PATH}.")
    except Exception as e:
        logging.error(f"Error loading employee DB: {e}")
    return {}

def draw_text_with_background(frame: np.ndarray, text: str, position: Tuple[int, int], font_scale: float = 0.5, text_color: Tuple[int, int, int] = (255, 255, 255), bg_color: Tuple[int, int, int] = (0, 0, 0), padding: int = 5):
    """Draws text with a semi-transparent background on a frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, 1)
    x, y = position
    bg_x1, bg_y1 = x - padding, y - text_h - padding
    bg_x2, bg_y2 = x + text_w + padding, y + padding
    overlay = frame.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, 1, cv2.LINE_AA)

def send_email_alert(config: argparse.Namespace, image_path: Path, alert_subject: str, alert_body: str, receiver_email: str):
    """Constructs and sends an email with an image attachment."""
    if not all([config.sender_email, config.email_password, receiver_email]):
        logging.warning("Email not sent. Missing sender/password/receiver details.")
        return
    message = MIMEMultipart()
    message["From"], message["To"], message["Subject"] = config.sender_email, receiver_email, alert_subject
    message.attach(MIMEText(alert_body, "plain"))
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
            server.sendmail(config.sender_email, receiver_email, message.as_string())
        logging.info(f"Email alert sent successfully to {receiver_email}")
    except Exception as e:
        logging.error(f"Failed to send email to {receiver_email}: {e}")

def run_inference(config: argparse.Namespace):
    """Main loop for video capture, inference, and alert handling."""
    db_conn = setup_database()
    
    Path(ALERT_IMAGE_DIR).mkdir(parents=True, exist_ok=True)
    logging.info(f"Alert images will be saved to '{ALERT_IMAGE_DIR}' directory.")

    employee_db = load_employee_db()
    try:
        model = YOLO(config.model_path)
    except Exception as e:
        logging.error(f"Failed to load YOLO model: {e}"); return
    cap = cv2.VideoCapture(config.source)
    if not cap.isOpened():
        logging.error(f"Cannot access video source '{config.source}'."); return
    logging.info(f"Video source '{config.source}' opened. Press 'q' to exit.")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    alert_trigger_start_time: Optional[float] = None
    compliance_trigger_start_time: Optional[float] = None
    last_violation_time: float = 0
    last_compliance_time: float = 0
    email_sent_confirmation_time: float = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("End of video stream."); break

            person_detected = False
            current_violations = set()
            equipped_items = set()
            results = model(frame, verbose=False)

            for result in results:
                for box in result.boxes:
                    class_name = model.names.get(int(box.cls[0]), "Unknown")
                    if class_name == "Person": person_detected = True
                    if class_name.startswith("NO-"): current_violations.add(class_name.replace("NO-", ""))
                    elif class_name in EQUIPPABLE_ITEMS: equipped_items.add(class_name)
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    color = CLASS_COLORS.get(class_name, DEFAULT_COLOR)
                    label = f"{class_name} {float(box.conf[0]):.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    draw_text_with_background(frame, label, (x1, y1 - 10), bg_color=color)

            y_pos = frame.shape[0] - 20
            if equipped_items:
                equipped_str = "Equipped: " + ", ".join(sorted(list(equipped_items)))
                draw_text_with_background(frame, equipped_str, (10, y_pos - 30), font_scale=0.6, text_color=(255, 255, 255), bg_color=(0, 128, 0))

            is_alert_condition = person_detected and len(current_violations) > 0
            is_compliant_condition = person_detected and len(current_violations) == 0 and len(equipped_items) > 0

            if is_alert_condition:
                compliance_trigger_start_time = None
                if alert_trigger_start_time is None: alert_trigger_start_time = time.time()
                elapsed_time = time.time() - alert_trigger_start_time
                violations_str = ", ".join(sorted(list(current_violations)))
                alert_text = f"ALERT: Missing {violations_str} ({int(elapsed_time)}s)"
                draw_text_with_background(frame, alert_text, (10, y_pos), font_scale=0.6, text_color=(255, 255, 255), bg_color=(0, 0, 200))

                if elapsed_time >= ALERT_THRESHOLD_SECONDS and (time.time() - last_violation_time) >= EMAIL_COOLDOWN_SECONDS:
                    
                    img_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_path = Path(ALERT_IMAGE_DIR) / f"violation_{img_timestamp}.jpg"
                    
                    cv2.imwrite(str(image_path), frame)
                    draw_text_with_background(frame, "PAUSED - Check Console", (10, 100), font_scale=0.7, text_color=(0, 255, 255), bg_color=(0, 0, 0))
                    cv2.imshow(WINDOW_NAME, frame)
                    cv2.waitKey(1)
                    logging.warning(f"VIOLATION DETECTED: Missing {violations_str}")
                    employee_id = print_big_input_prompt(["VIOLATION DETECTED", "Enter Employee ID (or press ENTER to skip)"])

                    if employee_id:
                        receiver_info = employee_db.get(employee_id)
                        
                        if receiver_info:
                            receiver_name = receiver_info.get("name", "Employee")
                            receiver_email = receiver_info.get("email")

                            if receiver_email:
                                alert_subject = f"Alert: PPE Violation for {receiver_name}"
                                alert_body = f"Hello {receiver_name},\n\nA PPE violation ({violations_str}) was detected...\n\nPlease review the attached image."
                                email_thread = threading.Thread(target=send_email_alert, args=(config, image_path, alert_subject, alert_body, receiver_email))
                                email_thread.start(); print("Sending email..."); email_thread.join()
                                email_sent_confirmation_time = time.time()
                                print(f"Alert sent to {receiver_name} ({receiver_email}).")
                                print_big_red_banner([f"VIOLATION LOGGED for ID: {employee_id}"])
                            else:
                                print_big_red_banner([f"VIOLATION LOGGED for ID: {employee_id}", f"No email on file for {receiver_name}."])
                            
                            # <-- MODIFICATION: Pass 'NO' for attendance
                            log_event_to_db(db_conn, employee_id, receiver_name, 'VIOLATION', 'NO', details=violations_str, image_path=image_path)
                        
                        else:
                            receiver_name = "Unknown ID"
                            print_big_red_banner([f"VIOLATION LOGGED for ID: {employee_id}", "ID NOT FOUND in employee database."])
                            
                            # <-- MODIFICATION: Pass 'NO' for attendance
                            log_event_to_db(db_conn, employee_id, receiver_name, 'VIOLATION', 'NO', details=violations_str, image_path=image_path)
                    
                    else:
                        print("Skipping alert."); print_big_red_banner(["OPERATOR SKIPPED"])
                    
                    print_big_yellow_prompt()
                    while input().strip() != '1': continue
                    alert_trigger_start_time = None
                    last_violation_time = 0 
                    print("Resuming detection...")

            elif is_compliant_condition:
                alert_trigger_start_time = None
                if compliance_trigger_start_time is None: compliance_trigger_start_time = time.time()
                elapsed_time = time.time() - compliance_trigger_start_time
                
                if elapsed_time >= ALERT_THRESHOLD_SECONDS and (time.time() - last_compliance_time) >= COMPLIANCE_COOLDOWN_SECONDS:
                    draw_text_with_background(frame, "Enter ID for ATTENDANCE !!", (int(frame.shape[1]*0.25), 100), font_scale=1.5, text_color=(255, 255, 255), bg_color=(0, 180, 0))
                    cv2.imshow(WINDOW_NAME, frame)
                    cv2.waitKey(1)
                    logging.info("COMPLIANCE DETECTED")
                    employee_id = print_big_input_prompt(["COMPLIANCE DETECTED", "Enter Employee ID to log attendance"])

                    if employee_id:
                        if employee_id in employee_db:
                            employee_name = employee_db[employee_id].get("name", "Unknown")
                            print_big_green_banner([f"ATTENDANCE LOGGED for ID: {employee_id}", f"Name: {employee_name}"])
                            # <-- MODIFICATION: Pass 'YES' for attendance
                            log_event_to_db(db_conn, employee_id, employee_name, 'COMPLIANCE', 'YES')
                        else:
                            employee_name = "Unknown ID"
                            print_big_red_banner([f"ATTENDANCE LOGGED for ID: {employee_id}", "ID NOT FOUND in employee database."])
                            # <-- MODIFICATION: Pass 'YES' for attendance
                            log_event_to_db(db_conn, employee_id, employee_name, 'COMPLIANCE', 'YES')
                    else:
                        print("Skipping attendance log."); print_big_red_banner(["ATTENDANCE NOT LOGGED", "OPERATOR SKIPPED"])
                    
                    print_big_yellow_prompt()
                    while input().strip() != '1': continue
                    compliance_trigger_start_time = None
                    last_compliance_time = 0
                    print("Resuming detection...")
            else:
                alert_trigger_start_time = None
                compliance_trigger_start_time = None

            if time.time() - email_sent_confirmation_time < EMAIL_SENT_MSG_DURATION_SECONDS:
                draw_text_with_background(frame, "Email Sent!", (frame.shape[1] - 150, 30), text_color=(0, 255, 0), bg_color=(0, 0, 0))
            
            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        if db_conn:
            db_conn.close()
            logging.info("Database connection closed.")
        cap.release()
        cv2.destroyAllWindows()
        logging.info("Resources released and windows closed.")

def main():
    """Main entry point of the script."""
    config = setup_config()
    run_inference(config)

if __name__ == "__main__":
    main()