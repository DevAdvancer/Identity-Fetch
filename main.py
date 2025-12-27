import customtkinter as ctk
import tkinter.messagebox as msg
from tkinter import filedialog, simpledialog
import cv2
import face_recognition
import pandas as pd
import numpy as np
import os
from PIL import Image
from datetime import datetime
import json
import uuid
import time
import threading


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

MASTER_ADMIN_PIN = "1234"
APP_WIDTH, APP_HEIGHT = 1000, 750
DB_FILE = "database.xlsx"
IMG_DIR = "images/gallery"
LOG_DIR = "images/unidentified_logs"
ICON_DIR = "icons"
BG_IMAGE_PATH = "background.jpeg"

# Performance settings
FRAME_RESIZE_SCALE = 0.25  # Resize to 25% for faster processing
PROCESS_EVERY_N_FRAMES = 3  # Process every 3rd frame
FACE_DETECTION_MODEL = "hog"  # "hog" is faster, "cnn" is more accurate

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

FONT_TITLE = ("Segoe UI", 32, "bold")
FONT_BTN = ("Segoe UI", 14, "bold")

failed_attempts = 0
current_user_id = None

ICON_SIZE = (24, 24)
icons = {}
icon_files = ["register", "login", "admin", "edit", "save", "upload", "gallery", "users", "delete", "logout"]
for ic in icon_files:
    path = os.path.join(ICON_DIR, f"{ic}.png")
    icons[ic] = ctk.CTkImage(Image.open(path), size=ICON_SIZE) if os.path.exists(path) else None


def init_database():
    """Initialize database"""
    if not os.path.exists(DB_FILE):
        try:
            with pd.ExcelWriter(DB_FILE, engine='openpyxl') as w:
                pd.DataFrame(columns=["user_id", "name", "age", "phone", "dept", "face_encoding"]).to_excel(
                    w, sheet_name="users", index=False)
                pd.DataFrame(columns=["admin_id", "name", "age", "phone", "dept", "face_encoding"]).to_excel(
                    w, sheet_name="admins", index=False)
                pd.DataFrame(columns=["image_id", "user_id", "image_path"]).to_excel(
                    w, sheet_name="gallery", index=False)
            print("Database created successfully.")
        except Exception as e:
            msg.showerror("Database Error", f"Could not create database: {e}")


def save_all_sheets(df_u, df_a, df_g):
    try:
        with pd.ExcelWriter(DB_FILE, engine='openpyxl') as w:
            df_u.to_excel(w, sheet_name="users", index=False)
            df_a.to_excel(w, sheet_name="admins", index=False)
            df_g.to_excel(w, sheet_name="gallery", index=False)
    except PermissionError:
        msg.showerror("File Error", "Cannot save! Please close 'database.xlsx' if it is open.")


def auto_cleanup_logs(days=7):
    now = time.time()
    for f in os.listdir(LOG_DIR):
        f_path = os.path.join(LOG_DIR, f)
        if os.stat(f_path).st_mtime < now - (days * 86400):
            try:
                os.remove(f_path)
            except:
                pass


init_database()
auto_cleanup_logs()


def update_details_logic(uid, role="user", refresh_callback=None):
    id_col = "user_id" if role == "user" else "admin_id"

    df_u = pd.read_excel(DB_FILE, sheet_name="users")
    df_a = pd.read_excel(DB_FILE, sheet_name="admins")
    df_g = pd.read_excel(DB_FILE, sheet_name="gallery")

    df = df_u if role == "user" else df_a

    if str(uid) not in df[id_col].astype(str).values:
        msg.showerror("Error", "ID not found.")
        return

    field = simpledialog.askstring("Update", "What to update? (name / age / phone / dept):")
    if not field:
        return
    field = field.lower()

    if field not in ["name", "age", "phone", "dept"]:
        msg.showerror("Error", "Invalid field.")
        return

    new_val = simpledialog.askstring("Update", f"Enter new value for {field}:")
    if new_val is None:
        return

    df.loc[df[id_col].astype(str) == str(uid), field] = new_val

    if role == "user":
        save_all_sheets(df, df_a, df_g)
    else:
        save_all_sheets(df_u, df, df_g)

    msg.showinfo("Success", "Details updated!")
    if refresh_callback:
        refresh_callback()


def is_blurry(frame, threshold=50.0):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold, variance


def get_face_crop(frame, location):
    top, right, bottom, left = location
    h, w = frame.shape[:2]
    padding = 30
    return frame[max(0, top - padding):min(h, bottom + padding),
                 max(0, left - padding):min(w, right + padding)]


def resize_frame(frame, scale):
    """Resize frame for faster processing"""
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    return cv2.resize(frame, (width, height))


def scale_face_locations(face_locations, scale):
    """Scale face locations back to original size"""
    return [(int(top/scale), int(right/scale), int(bottom/scale), int(left/scale))
            for (top, right, bottom, left) in face_locations]


def initialize_camera(camera_index=0):
    """Initialize camera with optimized settings"""
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)

    if cap.isOpened():
        # Optimize camera settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Warmup
        for _ in range(5):
            cap.read()

        return cap
    return None


def release_camera(cap):
    """Release camera and close windows"""
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)


def capture_face():
    """Fast face capture with optimized processing"""
    cap = initialize_camera()

    if cap is None:
        msg.showerror("Camera Error", "Could not open camera.")
        return None

    final_encoding = None
    frame_count = 0
    cached_face_locs = []
    cached_status = "Looking for face..."
    cached_color = (0, 0, 255)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            frame_count += 1
            display_frame = frame.copy()

            # Process every N frames for speed
            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                # Resize for faster processing
                small_frame = resize_frame(frame, FRAME_RESIZE_SCALE)
                rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                try:
                    # Fast face detection on small frame
                    small_face_locs = face_recognition.face_locations(rgb_small, model=FACE_DETECTION_MODEL)
                    # Scale back to original size
                    cached_face_locs = scale_face_locations(small_face_locs, FRAME_RESIZE_SCALE)

                    # Update status
                    if len(cached_face_locs) == 0:
                        cached_status = "No Face - Look at camera"
                        cached_color = (0, 0, 255)
                    elif len(cached_face_locs) > 1:
                        cached_status = "Multiple Faces - Show one only"
                        cached_color = (0, 165, 255)
                    else:
                        blurry, _ = is_blurry(frame)
                        if blurry:
                            cached_status = "Blurry - Hold still"
                            cached_color = (0, 165, 255)
                        else:
                            cached_status = "Ready! Press SPACE to capture"
                            cached_color = (0, 255, 0)
                except:
                    pass

            # Draw using cached results (smooth display)
            for (top, right, bottom, left) in cached_face_locs:
                cv2.rectangle(display_frame, (left, top), (right, bottom), cached_color, 2)
                # Draw corner accents
                corner_len = 20
                cv2.line(display_frame, (left, top), (left + corner_len, top), cached_color, 3)
                cv2.line(display_frame, (left, top), (left, top + corner_len), cached_color, 3)
                cv2.line(display_frame, (right, top), (right - corner_len, top), cached_color, 3)
                cv2.line(display_frame, (right, top), (right, top + corner_len), cached_color, 3)
                cv2.line(display_frame, (left, bottom), (left + corner_len, bottom), cached_color, 3)
                cv2.line(display_frame, (left, bottom), (left, bottom - corner_len), cached_color, 3)
                cv2.line(display_frame, (right, bottom), (right - corner_len, bottom), cached_color, 3)
                cv2.line(display_frame, (right, bottom), (right, bottom - corner_len), cached_color, 3)

            # Status bar at top
            cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 40), (0, 0, 0), -1)
            cv2.putText(display_frame, cached_status, (10, 28),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, cached_color, 2)

            # Instructions at bottom
            cv2.rectangle(display_frame, (0, display_frame.shape[0]-30),
                         (display_frame.shape[1], display_frame.shape[0]), (0, 0, 0), -1)
            cv2.putText(display_frame, "SPACE: Capture | Q: Cancel",
                       (10, display_frame.shape[0] - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Face Capture", display_frame)

            key = cv2.waitKey(1) & 0xFF

            # SPACE to capture
            if key == 32:  # Space bar
                if cached_color == (0, 255, 0) and len(cached_face_locs) == 1:
                    rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    try:
                        encs = face_recognition.face_encodings(rgb_full, cached_face_locs)
                        if encs:
                            final_encoding = encs[0]
                            break
                    except Exception as e:
                        print(f"Encoding error: {e}")

            # Q or ESC to quit
            elif key == ord('q') or key == ord('Q') or key == 27:
                break

    finally:
        release_camera(cap)

    return final_encoding


def upload_face(uid):
    file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
    if not file_path:
        return None, None

    frame = cv2.imread(file_path)
    if frame is None:
        msg.showerror("Error", "Could not read image file.")
        return None, None

    blurry, _ = is_blurry(frame)
    if blurry:
        msg.showerror("Error", "Image is too blurry.")
        return None, None

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    try:
        face_locs = face_recognition.face_locations(rgb)

        if len(face_locs) == 1:
            cropped = get_face_crop(frame, face_locs[0])
            img_id = datetime.now().strftime("%Y%m%d%H%M%S")
            save_path = os.path.join(IMG_DIR, f"{uid}_{img_id}.jpg")
            cv2.imwrite(save_path, cropped)
            encs = face_recognition.face_encodings(rgb, face_locs)
            if encs:
                return encs[0], save_path
        elif len(face_locs) == 0:
            msg.showerror("Error", "No face detected in image.")
        else:
            msg.showerror("Error", "Multiple faces detected. Use image with ONE face.")
    except Exception as e:
        msg.showerror("Error", f"Processing failed: {e}")

    return None, None


def register_logic(role="user", mode="camera"):
    if role == "admin":
        pin = simpledialog.askstring("Manager Access", "Enter PIN:", show='*')
        if pin != MASTER_ADMIN_PIN:
            msg.showerror("Error", "Incorrect PIN")
            return

    name = entry_name.get().strip()
    if not name:
        msg.showerror("Error", "Name is required")
        return

    prefix = "USR-" if role == "user" else "ADM-"
    uid = prefix + uuid.uuid4().hex[:6].upper()

    df_u = pd.read_excel(DB_FILE, sheet_name="users")
    df_a = pd.read_excel(DB_FILE, sheet_name="admins")
    df_g = pd.read_excel(DB_FILE, sheet_name="gallery")

    path = None
    if mode == "camera":
        face_enc = capture_face()
    else:
        face_enc, path = upload_face(uid)

    if face_enc is None:
        return

    enc_json = json.dumps(face_enc.tolist())

    if role == "user":
        new_row = {
            "user_id": uid,
            "name": name,
            "age": entry_age.get(),
            "phone": entry_phone.get(),
            "dept": entry_dept.get(),
            "face_encoding": enc_json
        }
        df_u = pd.concat([df_u, pd.DataFrame([new_row])], ignore_index=True)

        if path:
            df_g = pd.concat([df_g, pd.DataFrame([{
                "image_id": datetime.now().strftime("%f"),
                "user_id": uid,
                "image_path": path
            }])], ignore_index=True)
    else:
        new_row = {
            "admin_id": uid,
            "name": name,
            "age": entry_age.get(),
            "phone": entry_phone.get(),
            "dept": entry_dept.get(),
            "face_encoding": enc_json
        }
        df_a = pd.concat([df_a, pd.DataFrame([new_row])], ignore_index=True)

    save_all_sheets(df_u, df_a, df_g)

    # Clear form
    entry_name.delete(0, 'end')
    entry_age.delete(0, 'end')
    entry_phone.delete(0, 'end')
    entry_dept.delete(0, 'end')

    msg.showinfo("Success", f"Registration Complete!\n\nYour ID: {uid}")
    app.show_login()


def identify_face(df_u, df_a):
    """Fast face identification"""
    cap = initialize_camera()
    if cap is None:
        return None, None

    # Pre-load encodings for speed
    user_encodings = []
    user_ids = []
    admin_encodings = []
    admin_ids = []

    for _, row in df_u.iterrows():
        try:
            enc = np.array(json.loads(row["face_encoding"]))
            user_encodings.append(enc)
            user_ids.append(row["user_id"])
        except:
            continue

    for _, row in df_a.iterrows():
        try:
            enc = np.array(json.loads(row["face_encoding"]))
            admin_encodings.append(enc)
            admin_ids.append(row["admin_id"])
        except:
            continue

    found_id, found_role = None, None
    frame_count = 0
    cached_face_locs = []
    last_frame = None
    start_time = time.time()

    try:
        while True:
            # Timeout after 20 seconds
            if time.time() - start_time > 20:
                break

            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            frame_count += 1
            display_frame = frame.copy()
            last_frame = frame.copy()

            # Process every N frames
            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                small_frame = resize_frame(frame, FRAME_RESIZE_SCALE)
                rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                try:
                    small_face_locs = face_recognition.face_locations(rgb_small, model=FACE_DETECTION_MODEL)
                    cached_face_locs = scale_face_locations(small_face_locs, FRAME_RESIZE_SCALE)

                    if len(cached_face_locs) == 1:
                        rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        encs = face_recognition.face_encodings(rgb_full, cached_face_locs)

                        if encs:
                            # Check admins first
                            if admin_encodings:
                                matches = face_recognition.compare_faces(admin_encodings, encs[0], 0.45)
                                if True in matches:
                                    found_id = admin_ids[matches.index(True)]
                                    found_role = "admin"
                                    break

                            # Check users
                            if user_encodings:
                                matches = face_recognition.compare_faces(user_encodings, encs[0], 0.45)
                                if True in matches:
                                    found_id = user_ids[matches.index(True)]
                                    found_role = "user"
                                    break
                except:
                    pass

            # Draw
            for (top, right, bottom, left) in cached_face_locs:
                cv2.rectangle(display_frame, (left, top), (right, bottom), (255, 255, 0), 2)

            # Scanning animation
            scan_y = int((frame_count % 60) / 60 * display_frame.shape[0])
            cv2.line(display_frame, (0, scan_y), (display_frame.shape[1], scan_y), (0, 255, 255), 1)

            cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 40), (0, 0, 0), -1)
            cv2.putText(display_frame, "Scanning... Look at camera", (10, 28),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.rectangle(display_frame, (0, display_frame.shape[0]-30),
                         (display_frame.shape[1], display_frame.shape[0]), (0, 0, 0), -1)
            cv2.putText(display_frame, "Q: Cancel", (10, display_frame.shape[0] - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Identifying", display_frame)

            if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q'), 27]:
                break

    finally:
        release_camera(cap)

    return found_id, found_role, last_frame


def login_logic():
    app.withdraw()
    global failed_attempts, current_user_id

    df_u = pd.read_excel(DB_FILE, sheet_name="users")
    df_a = pd.read_excel(DB_FILE, sheet_name="admins")

    if df_u.empty and df_a.empty:
        msg.showwarning("No Users", "No users registered yet. Please register first.")
        app.deiconify()
        return

    result = identify_face(df_u, df_a)

    if result[0]:
        found_id, found_role, _ = result
        failed_attempts = 0
        current_user_id = found_id

        if found_role == "admin":
            admin_profile_page()
        else:
            user_profile_page()
    else:
        app.deiconify()
        failed_attempts += 1

        if failed_attempts >= 3:
            # Log breach attempt
            if result[2] is not None:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(os.path.join(LOG_DIR, f"breach_{ts}.jpg"), result[2])

            msg.showerror("Security Alert", "3 failed attempts. Incident logged.")
            failed_attempts = 0
        else:
            msg.showwarning("Not Recognized", f"Face not recognized (Attempt {failed_attempts}/3)")


def admin_login_logic():
    pin = simpledialog.askstring("Manager Access", "Enter PIN:", show='*')
    if pin != MASTER_ADMIN_PIN:
        msg.showerror("Error", "Incorrect PIN")
        return

    app.withdraw()
    global current_user_id

    df_a = pd.read_excel(DB_FILE, sheet_name="admins")

    if df_a.empty:
        msg.showwarning("No Managers", "No managers registered. Please register first.")
        app.deiconify()
        return

    # Create dummy df_u for identify_face
    df_u = pd.DataFrame(columns=["user_id", "face_encoding"])

    result = identify_face(df_u, df_a)

    if result[0] and result[1] == "admin":
        current_user_id = result[0]
        admin_profile_page()
    else:
        app.deiconify()
        msg.showwarning("Access Denied", "Manager face not recognized.")


def create_sidebar(parent, role):
    sidebar = ctk.CTkFrame(parent, width=200, corner_radius=0, fg_color="#1a1a2e")
    sidebar.pack(side="left", fill="y")
    sidebar.pack_propagate(False)

    # Logo/Title
    ctk.CTkLabel(sidebar, text="MENU", font=("Segoe UI", 20, "bold"),
                text_color="#00d4ff").pack(pady=30)

    # Navigation buttons
    btn_style = {"width": 180, "height": 45, "corner_radius": 10, "font": ("Segoe UI", 13)}

    ctk.CTkButton(sidebar, text="  Profile", image=icons.get("users"), compound="left",
                  fg_color="#16213e", hover_color="#0f3460",
                  command=lambda: switch_page(parent, admin_profile_page if role == "admin" else user_profile_page),
                  **btn_style).pack(pady=5)

    if role == "admin":
        ctk.CTkButton(sidebar, text="  Dashboard", image=icons.get("users"), compound="left",
                      fg_color="#16213e", hover_color="#0f3460",
                      command=lambda: switch_page(parent, admin_dashboard_page),
                      **btn_style).pack(pady=5)

        ctk.CTkButton(sidebar, text="  Breach Logs", image=icons.get("gallery"), compound="left",
                      fg_color="#16213e", hover_color="#0f3460",
                      command=lambda: view_logs_popup(parent),
                      **btn_style).pack(pady=5)

    # Logout at bottom
    ctk.CTkButton(sidebar, text="  Logout", image=icons.get("logout"), compound="left",
                  fg_color="#e74c3c", hover_color="#c0392b",
                  command=lambda: [parent.destroy(), app.deiconify()],
                  **btn_style).pack(side="bottom", pady=30)


def switch_page(current_win, new_page_func):
    current_win.destroy()
    new_page_func()


def view_logs_popup(parent):
    popup = ctk.CTkToplevel(parent)
    popup.geometry("600x500")
    popup.title("Breach Logs")
    popup.transient(parent)

    scroll = ctk.CTkScrollableFrame(popup, width=560, height=400)
    scroll.pack(pady=20, padx=20, fill="both", expand=True)

    log_files = sorted(os.listdir(LOG_DIR), reverse=True)

    if not log_files:
        ctk.CTkLabel(scroll, text="No breach attempts logged.",
                    font=("Segoe UI", 14)).pack(pady=50)
    else:
        for log_file in log_files:
            log_path = os.path.join(LOG_DIR, log_file)

            frame = ctk.CTkFrame(scroll, fg_color="#2d2d2d")
            frame.pack(fill="x", pady=3)

            timestamp = log_file.replace("breach_", "").replace("unknown_", "").replace(".jpg", "")
            ctk.CTkLabel(frame, text=timestamp, font=("Segoe UI", 11)).pack(side="left", padx=15, pady=10)

            try:
                img = Image.open(log_path).resize((60, 45))
                photo = ctk.CTkImage(img, size=(60, 45))
                ctk.CTkLabel(frame, image=photo, text="").pack(side="left", padx=10)
            except:
                pass

            ctk.CTkButton(frame, text="Delete", fg_color="#e74c3c", width=70, height=30,
                         command=lambda p=log_path, f=frame: [os.remove(p) if os.path.exists(p) else None, f.destroy()]
                         ).pack(side="right", padx=10, pady=5)


def user_profile_page():
    df_u = pd.read_excel(DB_FILE, sheet_name="users")
    df_g = pd.read_excel(DB_FILE, sheet_name="gallery")

    user_data = df_u[df_u["user_id"].astype(str) == str(current_user_id)]
    if user_data.empty:
        msg.showerror("Error", "User not found")
        app.deiconify()
        return

    user = user_data.iloc[0]

    win = ctk.CTkToplevel()
    win.geometry("950x650")
    win.title(f"Profile - {user['name']}")

    create_sidebar(win, "user")

    # Main content
    main = ctk.CTkFrame(win, fg_color="transparent")
    main.pack(side="right", fill="both", expand=True, padx=40, pady=40)

    # Profile card
    card = ctk.CTkFrame(main, corner_radius=20, fg_color="#1e1e2e")
    card.pack(fill="both", expand=True)

    # Profile image
    user_img_path = None
    gallery_matches = df_g[df_g["user_id"].astype(str) == str(current_user_id)]
    if not gallery_matches.empty:
        user_img_path = gallery_matches.iloc[-1]["image_path"]

    if user_img_path and os.path.exists(str(user_img_path)):
        try:
            img = Image.open(user_img_path).resize((150, 150))
            profile_img = ctk.CTkImage(img, size=(150, 150))
            img_label = ctk.CTkLabel(card, image=profile_img, text="")
            img_label.pack(pady=30)
        except:
            pass

    # Name
    ctk.CTkLabel(card, text=user['name'], font=("Segoe UI", 28, "bold")).pack(pady=10)

    # Details
    details = ctk.CTkFrame(card, fg_color="transparent")
    details.pack(pady=20)

    for label, key in [("ID", "user_id"), ("Department", "dept"), ("Phone", "phone"), ("Age", "age")]:
        row = ctk.CTkFrame(details, fg_color="transparent")
        row.pack(fill="x", pady=8)
        ctk.CTkLabel(row, text=f"{label}:", font=("Segoe UI", 13, "bold"),
                    width=120, anchor="e", text_color="#888").pack(side="left")
        ctk.CTkLabel(row, text=str(user[key]), font=("Segoe UI", 13),
                    width=200, anchor="w").pack(side="left", padx=20)

    # Edit button
    ctk.CTkButton(card, text="Edit Details", image=icons.get("edit"), compound="left",
                  width=200, height=45, corner_radius=22,
                  command=lambda: update_details_logic(current_user_id, refresh_callback=lambda: switch_page(win, user_profile_page))
                  ).pack(pady=30)


def admin_profile_page():
    df_a = pd.read_excel(DB_FILE, sheet_name="admins")

    admin_data = df_a[df_a["admin_id"].astype(str) == str(current_user_id)]
    if admin_data.empty:
        msg.showerror("Error", "Admin not found")
        app.deiconify()
        return

    admin = admin_data.iloc[0]

    win = ctk.CTkToplevel()
    win.geometry("950x650")
    win.title(f"Manager - {admin['name']}")

    create_sidebar(win, "admin")

    main = ctk.CTkFrame(win, fg_color="transparent")
    main.pack(side="right", fill="both", expand=True, padx=40, pady=40)

    card = ctk.CTkFrame(main, corner_radius=20, fg_color="#1e1e2e")
    card.pack(fill="both", expand=True)

    # Manager badge
    ctk.CTkLabel(card, text="👔", font=("Segoe UI", 60)).pack(pady=20)
    ctk.CTkLabel(card, text=admin['name'], font=("Segoe UI", 28, "bold")).pack(pady=5)
    ctk.CTkLabel(card, text="MANAGER", font=("Segoe UI", 12), text_color="#00d4ff").pack()

    details = ctk.CTkFrame(card, fg_color="transparent")
    details.pack(pady=30)

    for label, key in [("ID", "admin_id"), ("Department", "dept"), ("Phone", "phone"), ("Age", "age")]:
        row = ctk.CTkFrame(details, fg_color="transparent")
        row.pack(fill="x", pady=8)
        ctk.CTkLabel(row, text=f"{label}:", font=("Segoe UI", 13, "bold"),
                    width=120, anchor="e", text_color="#888").pack(side="left")
        ctk.CTkLabel(row, text=str(admin[key]), font=("Segoe UI", 13),
                    width=200, anchor="w").pack(side="left", padx=20)

    btn_frame = ctk.CTkFrame(card, fg_color="transparent")
    btn_frame.pack(pady=20)

    ctk.CTkButton(btn_frame, text="Edit", image=icons.get("edit"), compound="left",
                  width=150, height=40, corner_radius=20,
                  command=lambda: update_details_logic(current_user_id, "admin",
                           refresh_callback=lambda: switch_page(win, admin_profile_page))
                  ).pack(side="left", padx=10)

    ctk.CTkButton(btn_frame, text="Dashboard", image=icons.get("users"), compound="left",
                  width=150, height=40, corner_radius=20, fg_color="#16213e",
                  command=lambda: switch_page(win, admin_dashboard_page)
                  ).pack(side="left", padx=10)


def admin_dashboard_page():
    win = ctk.CTkToplevel()
    win.geometry("1150x750")
    win.title("Admin Dashboard")

    create_sidebar(win, "admin")

    main = ctk.CTkFrame(win, fg_color="transparent")
    main.pack(side="right", fill="both", expand=True, padx=20, pady=20)

    # Tabs
    tabview = ctk.CTkTabview(main, width=850, height=650)
    tabview.pack(fill="both", expand=True)

    tab_users = tabview.add("Users")
    tab_admins = tabview.add("Managers")
    tab_logs = tabview.add("Security Logs")

    # ===== USERS TAB =====
    txt_users = ctk.CTkTextbox(tab_users, width=800, height=450, font=("Consolas", 11))
    txt_users.pack(pady=15)

    def refresh_users():
        df = pd.read_excel(DB_FILE, sheet_name="users")
        txt_users.configure(state="normal")
        txt_users.delete("0.0", "end")
        display_df = df.drop(columns=["face_encoding"], errors="ignore")
        txt_users.insert("0.0", display_df.to_string(index=False) if not display_df.empty else "No users registered.")
        txt_users.configure(state="disabled")

    btn_frame = ctk.CTkFrame(tab_users, fg_color="transparent")
    btn_frame.pack(pady=10)

    def delete_user():
        uid = simpledialog.askstring("Delete User", "Enter User ID:")
        if uid and msg.askyesno("Confirm", f"Delete user {uid}?"):
            df_u = pd.read_excel(DB_FILE, sheet_name="users")
            df_a = pd.read_excel(DB_FILE, sheet_name="admins")
            df_g = pd.read_excel(DB_FILE, sheet_name="gallery")

            # Delete images
            for img_p in df_g[df_g["user_id"].astype(str) == str(uid)]["image_path"].values:
                if os.path.exists(str(img_p)):
                    try: os.remove(img_p)
                    except: pass

            df_u = df_u[df_u["user_id"].astype(str) != str(uid)]
            df_g = df_g[df_g["user_id"].astype(str) != str(uid)]
            save_all_sheets(df_u, df_a, df_g)
            refresh_users()
            msg.showinfo("Done", f"User {uid} deleted.")

    ctk.CTkButton(btn_frame, text="Refresh", width=120, command=refresh_users).pack(side="left", padx=5)
    ctk.CTkButton(btn_frame, text="Update", width=120,
                  command=lambda: update_details_logic(simpledialog.askstring("Update", "Enter User ID:"),
                           refresh_callback=refresh_users)).pack(side="left", padx=5)
    ctk.CTkButton(btn_frame, text="Delete", width=120, fg_color="#e74c3c",
                  command=delete_user).pack(side="left", padx=5)

    # ===== ADMINS TAB =====
    txt_admins = ctk.CTkTextbox(tab_admins, width=800, height=450, font=("Consolas", 11))
    txt_admins.pack(pady=15)

    def refresh_admins():
        df = pd.read_excel(DB_FILE, sheet_name="admins")
        txt_admins.configure(state="normal")
        txt_admins.delete("0.0", "end")
        display_df = df.drop(columns=["face_encoding"], errors="ignore")
        txt_admins.insert("0.0", display_df.to_string(index=False) if not display_df.empty else "No managers registered.")
        txt_admins.configure(state="disabled")

    btn_frame_a = ctk.CTkFrame(tab_admins, fg_color="transparent")
    btn_frame_a.pack(pady=10)

    ctk.CTkButton(btn_frame_a, text="Refresh", width=120, command=refresh_admins).pack(side="left", padx=5)
    ctk.CTkButton(btn_frame_a, text="Update", width=120,
                  command=lambda: update_details_logic(simpledialog.askstring("Update", "Enter Manager ID:"),
                           role="admin", refresh_callback=refresh_admins)).pack(side="left", padx=5)

    # ===== LOGS TAB =====
    log_box = ctk.CTkTextbox(tab_logs, width=800, height=450, font=("Consolas", 11))
    log_box.pack(pady=15)

    def refresh_logs():
        files = sorted(os.listdir(LOG_DIR), reverse=True)
        log_box.configure(state="normal")
        log_box.delete("0.0", "end")
        if files:
            for f in files:
                log_box.insert("end", f"{f}\n")
        else:
            log_box.insert("end", "No breach attempts logged.")
        log_box.configure(state="disabled")

    def clear_logs():
        if msg.askyesno("Confirm", "Delete all logs?"):
            for f in os.listdir(LOG_DIR):
                try: os.remove(os.path.join(LOG_DIR, f))
                except: pass
            refresh_logs()

    btn_frame_l = ctk.CTkFrame(tab_logs, fg_color="transparent")
    btn_frame_l.pack(pady=10)

    ctk.CTkButton(btn_frame_l, text="Refresh", width=120, command=refresh_logs).pack(side="left", padx=5)
    ctk.CTkButton(btn_frame_l, text="Clear All", width=120, fg_color="#e74c3c",
                  command=clear_logs).pack(side="left", padx=5)

    # Initial load
    refresh_users()
    refresh_admins()
    refresh_logs()


def safe_call(func):
    try:
        func()
    except Exception as e:
        import traceback
        traceback.print_exc()
        msg.showerror("Error", str(e))
        app.deiconify()


class FaceApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Identity Fetch AI")
        self.geometry(f"{APP_WIDTH}x{APP_HEIGHT}")
        self.resizable(False, False)

        # Background
        self.bg_image = None
        if os.path.exists(BG_IMAGE_PATH):
            try:
                self.bg_image = ctk.CTkImage(Image.open(BG_IMAGE_PATH), size=(APP_WIDTH, APP_HEIGHT))
                self.bg_label = ctk.CTkLabel(self, image=self.bg_image, text="")
                self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)
            except:
                pass

        self.main_container = ctk.CTkFrame(self, fg_color="transparent" if self.bg_image else "gray10")
        self.main_container.place(relx=0, rely=0, relwidth=1, relheight=1)

        self.setup_nav()
        self.setup_forms()
        self.show_login()

    def setup_nav(self):
        nav = ctk.CTkFrame(self.main_container, fg_color="transparent", height=80)
        nav.pack(side="top", fill="x", padx=50, pady=30)

        ctk.CTkLabel(nav, text="IDENTITY FETCH AI", font=("Segoe UI", 26, "bold"),
                    text_color="white").pack(side="left")

        btn_box = ctk.CTkFrame(nav, fg_color="transparent")
        btn_box.pack(side="right")

        ctk.CTkButton(btn_box, text="Identify", width=120, height=40, corner_radius=20,
                     fg_color="#2d2d2d", hover_color="#404040",
                     command=self.show_login).pack(side="left", padx=5)
        ctk.CTkButton(btn_box, text="Register", width=120, height=40, corner_radius=20,
                     fg_color="#2d2d2d", hover_color="#404040",
                     command=self.show_register).pack(side="left", padx=5)

    def setup_forms(self):
        self.card_container = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.card_container.place(relx=0.5, rely=0.55, anchor="center")

        # === LOGIN CARD ===
        self.login_card = ctk.CTkFrame(self.card_container, fg_color="transparent")

        ctk.CTkLabel(self.login_card, text="Verify Identity", font=FONT_TITLE,
                    text_color="white").pack(pady=(0, 40))

        ctk.CTkButton(self.login_card, text="  Start Face Scan", image=icons.get("login"),
                      compound="left", width=420, height=60, corner_radius=30, font=FONT_BTN,
                      fg_color="#0066cc", hover_color="#0052a3",
                      command=lambda: safe_call(login_logic)).pack(pady=12)

        ctk.CTkButton(self.login_card, text="  Manager Login", image=icons.get("admin"),
                      compound="left", width=420, height=60, corner_radius=30, font=FONT_BTN,
                      fg_color="#6b21a8", hover_color="#581c87",
                      command=lambda: safe_call(admin_login_logic)).pack(pady=12)

        # === REGISTER CARD ===
        self.reg_card = ctk.CTkFrame(self.card_container, fg_color="transparent")

        ctk.CTkLabel(self.reg_card, text="New Registration", font=FONT_TITLE,
                    text_color="white").pack(pady=(0, 25))

        global entry_name, entry_age, entry_phone, entry_dept
        field_style = {"width": 420, "height": 50, "corner_radius": 25,
                      "fg_color": "#1a1a2e", "border_width": 1, "border_color": "#333"}

        entry_name = ctk.CTkEntry(self.reg_card, placeholder_text="Full Name", **field_style)
        entry_age = ctk.CTkEntry(self.reg_card, placeholder_text="Age", **field_style)
        entry_phone = ctk.CTkEntry(self.reg_card, placeholder_text="Phone Number", **field_style)
        entry_dept = ctk.CTkEntry(self.reg_card, placeholder_text="Department", **field_style)

        for e in [entry_name, entry_age, entry_phone, entry_dept]:
            e.pack(pady=7)

        ctk.CTkButton(self.reg_card, text="  Capture with Camera", image=icons.get("register"),
                      compound="left", width=420, height=55, corner_radius=27, font=FONT_BTN,
                      fg_color="#0066cc", hover_color="#0052a3",
                      command=lambda: register_logic("user", "camera")).pack(pady=12)

        ctk.CTkButton(self.reg_card, text="  Upload Photo", image=icons.get("upload"),
                      compound="left", width=420, height=55, corner_radius=27, font=FONT_BTN,
                      fg_color="#8b4513", hover_color="#6b3410",
                      command=lambda: register_logic("user", "upload")).pack(pady=5)

        ctk.CTkButton(self.reg_card, text="  Register as Manager", image=icons.get("admin"),
                      compound="left", width=420, height=55, corner_radius=27, font=FONT_BTN,
                      fg_color="#6b21a8", hover_color="#581c87",
                      command=lambda: register_logic("admin", "camera")).pack(pady=12)

    def show_login(self):
        self.reg_card.pack_forget()
        self.login_card.pack()

    def show_register(self):
        self.login_card.pack_forget()
        self.reg_card.pack()


if __name__ == "__main__":
    print("=" * 50)
    print("IDENTITY FETCH AI")
    print("=" * 50)
    print("\nManager PIN: 1234")
    print("Controls:")
    print("  - SPACE: Capture face")
    print("  - Q: Cancel/Exit camera")
    print("=" * 50)

    app = FaceApp()
    app.mainloop()
