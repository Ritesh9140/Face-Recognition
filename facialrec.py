#!/usr/bin/env python3
import face_recognition
import cv2
import numpy as np
import csv
import argparse
from datetime import datetime, timedelta
import dlib
from scipy.spatial import distance
from imutils import face_utils
import os
import time

# ============================
# Global Configurations
# ============================
MIN_DURATION_MINUTES = 30      # minutes after entry before reappearance is valid

# ============================
# Utility: Map Name → Branch
# ============================
def get_branch(name):
    if "Ritesh" in name:
        return "AIDS"
    elif "Shashwat" in name:
        return "ECE"
    elif "Venketsh" in name or "Venky" in name:
        return "CSE"
    else:
        return "VLSI"

# ============================
# Blink/Liveness Setup
# ============================
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

EYE_AR_THRESH = 0.21
EYE_AR_CONSEC_FRAMES = 2

predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# ----------------------------
# Load Known Faces (once)
# ----------------------------
def load_known_faces():
    encodings, names = [], []
    for name, path in [
        ("Ritesh Singh", "faces/ritesh.jpg"),
        ("Shashwat Rao", "faces/shashwat.jpg"),
        ("V. Venketsh", "faces/venky.jpg")
    ]:
        img = face_recognition.load_image_file(path)
        e = face_recognition.face_encodings(img)
        if e:
            encodings.append(e[0])
            names.append(name)
    return encodings, names

global_known_encodings, global_known_names = load_known_faces()

# ----------------------------
# CSV Setup
# ----------------------------
def create_csv_file():
    today = datetime.now().strftime("%Y-%m-%d")
    fn = f"{today}.csv"
    with open(fn, "a+", newline="") as f:
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(["Name","Date","Entry Time","Check Time","Status","Branch"])
    return fn

# ----------------------------
# Record Attendance
# ----------------------------
def record_attendance(name, csv_fn, entry, check, status):
    branch = get_branch(name)
    date_str = entry.strftime("%Y-%m-%d")
    e_str = entry.strftime("%H:%M:%S")
    c_str = check.strftime("%H:%M:%S")
    with open(csv_fn, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([name, date_str, e_str, c_str, status, branch])
    print(f"→ {name}: {status} (entry {e_str}, check {c_str}), branch {branch}")
    # snapshot
    sd="snapshots"; os.makedirs(sd,exist_ok=True)
    fn=f"{name}_{date_str}_{e_str.replace(':','-')}_to_{c_str.replace(':','-')}.jpg"
    cv2.imwrite(os.path.join(sd,fn), last_frame)

# ----------------------------
# Compute Sessions 9:00–10:10, 10:10–11:10, … until 20:00
# ----------------------------
def compute_sessions():
    base = datetime.now().replace(hour=0,minute=0,second=0,microsecond=0)
    sessions=[]
    s=base+timedelta(hours=9); e=base+timedelta(hours=10,minutes=10)
    sessions.append((s,e))
    while e<base+timedelta(hours=20):
        s=e; e=s+timedelta(hours=1)
        if e>base+timedelta(hours=20): e=base+timedelta(hours=20)
        sessions.append((s,e))
    return sessions

# ----------------------------
# Process One Session
# ----------------------------
def process_session(start, end, cap, csv_fn):
    print(f"\n--- Session {start.strftime('%H:%M')}–{end.strftime('%H:%M')} ---")
    tracker={}  # name->{entry,reappeared,reappear_time}
    global last_frame
    last_frame=None

    while datetime.now()<end:
        ret, frame=cap.read()
        if not ret: continue
        last_frame=frame; now=datetime.now()

        # face detection
        small=cv2.resize(frame,(0,0),fx=0.5,fy=0.5)
        rgb=cv2.cvtColor(small,cv2.COLOR_BGR2RGB)
        locs=face_recognition.face_locations(rgb)
        encs=face_recognition.face_encodings(rgb,locs)

        seen=[]
        for fe,loc in zip(encs,locs):
            d=face_recognition.face_distance(global_known_encodings,fe)
            i=np.argmin(d); name="Unknown"
            if d[i]<0.4: name=global_known_names[i]
            seen.append(name)
            (t,r,b,l)=[v*2 for v in loc]
            cv2.rectangle(frame,(l,t),(r,b),(0,255,0),2)
            cv2.putText(frame,name,(l,t-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

            if name=="Unknown": continue
            if name not in tracker:
                tracker[name]={"entry":now,"reappeared":False,"reappear_time":None}
                print(f"{name} entry at {now.strftime('%H:%M:%S')}")
            else:
                data=tracker[name]
                if not data["reappeared"] and now>=data["entry"]+timedelta(minutes=MIN_DURATION_MINUTES):
                    data["reappeared"]=True
                    data["reappear_time"]=now
                    # take snapshot on reappearance
                    sd="snapshots"; os.makedirs(sd,exist_ok=True)
                    fn=f"{name}_{data['entry'].strftime('%H-%M-%S')}_reappear_{now.strftime('%H-%M-%S')}.jpg"
                    cv2.imwrite(os.path.join(sd,fn),frame)
                    print(f"🔔 snapshot reappear {name} at {now.strftime('%H:%M:%S')}")

        # show
        cv2.imshow("Attendance",frame)
        if cv2.waitKey(1)==ord('q'): break

    # session end: evaluate each
    for name,data in tracker.items():
        if data["reappeared"]:
            record_attendance(name,csv_fn,data["entry"],data["reappear_time"],"Present")
        else:
            record_attendance(name,csv_fn,data["entry"],end,"Absent")

# ----------------------------
# Run All Sessions
# ----------------------------
def run_day_sessions(cam_idx):
    csv_fn=create_csv_file()
    sessions=compute_sessions()
    cap=cv2.VideoCapture(cam_idx,cv2.CAP_DSHOW)
    for s,e in sessions:
        while datetime.now()<s: time.sleep(1)
        process_session(s,e,cap,csv_fn)
    cap.release(); cv2.destroyAllWindows()

# ============================
# Main + CLI
# ============================
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--min-duration",type=int,default=30)
    p.add_argument("--camera-index",type=int,default=1)
    args=p.parse_args()
    MIN_DURATION_MINUTES=args.min_duration
    run_day_sessions(args.camera_index)
