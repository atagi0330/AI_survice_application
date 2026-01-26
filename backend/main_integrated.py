import os
import sys
import time
import traceback
import datetime
import glob
import cv2
import numpy as np
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import warnings

# --- AI Libraries ---
from ultralytics import YOLO
import mediapipe as mp
import insightface
from insightface.app import FaceAnalysis

# --- 自作モジュール ---
import config
import features

# ログ抑制
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.simplefilter("ignore")

# ==========================================
# ★ 設定・定数 (顔認識・居眠り用)
# ==========================================
REC_THRESHOLD = 0.35
THRESHOLD_EAR_CLOSE = 0.13
THRESHOLD_EAR_HALF  = 0.20
TIME_SLEEP   = 2.0
TIME_UTOUTO  = 1.0
NOD_FRAME_WINDOW = 30 
NOD_THRESHOLD    = 0.15 
THRESHOLD_MAR = 0.45      
TIME_LIMIT_YAWN = 1.5   
SMOOTHING_WINDOW = 6

# MediaPipe Indices
IDX_L_EYE = [33, 160, 158, 133, 153, 144]
IDX_R_EYE = [362, 385, 387, 263, 373, 380]
IDX_MOUTH = [13, 14, 61, 291]
IDX_NOSE  = 1 

# 画像保存ワーカー
save_executor = ThreadPoolExecutor(max_workers=1)

# ==========================================
# クラス定義
# ==========================================
class FacePersonState:
    def __init__(self, pid):
        self.id = pid
        self.name = "Unknown"
        self.ear_buffer = deque(maxlen=SMOOTHING_WINDOW)
        self.mar_buffer = deque(maxlen=SMOOTHING_WINDOW)
        self.nose_y_buffer = deque(maxlen=NOD_FRAME_WINDOW)
        self.sleep_start = None
        self.utouto_start = None
        self.yawn_start = None
        self.status = "normal"
        self.msg = ""
        self.last_seen = time.time()
        self.bbox = [0, 0, 0, 0]

# ==========================================
# ユーティリティ関数
# ==========================================
def calc_ear(pts):
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    h = np.linalg.norm(pts[0] - pts[3])
    return (v1 + v2) / (2.0 * h) if h!=0 else 0

def calc_mar(pts):
    v = np.linalg.norm(pts[0] - pts[1])
    h = np.linalg.norm(pts[2] - pts[3])
    return v / h if h!=0 else 0

def compute_sim(feat1, feat2):
    return np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))

def imread_safe(filename):
    try:
        n = np.fromfile(filename, np.uint8)
        return cv2.imdecode(n, cv2.IMREAD_COLOR)
    except:
        return None

def load_known_faces(app, directory="face_db"):
    known_features = []
    known_names = []
    
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        return [], []

    files = glob.glob(os.path.join(directory, '*'))
    valid_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    print(f"顔DB読み込み中... ({len(valid_files)}枚)")

    for file_path in valid_files:
        basename = os.path.basename(file_path)
        img_org = imread_safe(file_path)
        if img_org is None: continue
        
        # 自動回転ロジック
        found = False
        img_temp = img_org.copy()
        for _ in range(4):
            faces = app.get(img_temp)
            if len(faces) > 0:
                faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
                known_features.append(faces[0].embedding)
                name, _ = os.path.splitext(basename)
                known_names.append(name)
                found = True
                break
            img_temp = cv2.rotate(img_temp, cv2.ROTATE_90_CLOCKWISE)
        
        if found:
            print(f" - 登録: {known_names[-1]}")

    return known_features, known_names

def save_snapshot_task(frame, mode):
    try:
        save_dir = os.path.join(config.LOG_DIR, mode)
        os.makedirs(save_dir, exist_ok=True)
        filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3] + ".jpg"
        cv2.imwrite(os.path.join(save_dir, filename), frame)
        # 古いファイル削除
        if mode == "normal":
            files = sorted(glob.glob(os.path.join(save_dir, "*.jpg")))
            if len(files) > config.NORMAL_MAX_FILES:
                try: os.remove(files[0])
                except: pass
    except: pass

# ==========================================
# メイン処理
# ==========================================
def main():
    print("=== AI統合システム (顔認証 + 転倒検知) 起動 ===")

    # 1. InsightFace 初期化
    print("Initialize InsightFace...")
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    known_feats, known_names = load_known_faces(app)

    # 2. MediaPipe 初期化
    print("Initialize MediaPipe...")
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.3)

    # 3. YOLO Pose 初期化
    print(f"Initialize YOLO Pose ({config.POSE_MODEL_PATH})...")
    model_path = config.POSE_MODEL_PATH
    if not os.path.exists(model_path):
        model_path = "yolov8n-pose.pt" # Fallback
    yolo_model = YOLO(model_path)
    yolo_model.to(config.DEVICE)

    # 4. カメラ初期化
    # ★重要: ここでカメラIDを指定 (0 または 1)
    # 映らない場合はここを 1 に変えてください
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
    if not cap.isOpened():
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("エラー: カメラが見つかりません。接続を確認してください。")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # 状態管理
    pose_people_states = {} 
    pose_last_seen = {}
    
    face_people_states = {}
    face_next_id = 0

    last_normal_save = time.time()
    last_alert_save = 0

    print("=== システム稼働開始 (終了は 'q' キー) ===")

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            current_time = time.time()
            h, w, _ = frame.shape
            
            # 結果を描画するメイン画像
            annotated_frame = frame.copy()
            trigger_alert_save = False

            # ---------------------------------------------------------
            # [A] YOLO Pose (転倒・ふらつき・姿勢)
            # ---------------------------------------------------------
            results = yolo_model.track(frame, persist=True, verbose=False, device=config.DEVICE, classes=[0])

            if results[0].boxes.id is not None:
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                keypoints_all = results[0].keypoints.data.cpu().numpy()
                boxes_all = results[0].boxes.xyxy.cpu().numpy()

                for i, track_id in enumerate(ids):
                    if track_id not in pose_people_states:
                        pose_people_states[track_id] = features.PersonState()
                    
                    pose_last_seen[track_id] = current_time
                    state = pose_people_states[track_id]
                    kpts = keypoints_all[i]
                    box = boxes_all[i]
                    x1, y1, x2, y2 = box.astype(int)

                    # 判定ロジック (features.py)
                    is_pose_fall = features.check_fall_pose(kpts, box)
                    is_bad_posture_now = features.check_bad_posture(kpts)
                    is_stagger_now = features.check_staggering(kpts, state)

                    state.update_status(is_pose_fall, is_stagger_now, is_bad_posture_now)
                    if state.alert_active: trigger_alert_save = True

                    # 描画 (全身枠)
                    color = state.status_color
                    # 枠線
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    # ラベル背景
                    label = f"ID:{track_id} {state.action_message}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(annotated_frame, (x1, y1 - 25), (x1 + tw, y1), color, -1)
                    # テキスト
                    cv2.putText(annotated_frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # ---------------------------------------------------------
            # [B] 顔認識 & 居眠り検知
            # ---------------------------------------------------------
            faces = app.get(frame)
            for face in faces:
                bbox = face.bbox.astype(int)
                embedding = face.embedding
                
                # 1. 顔認証
                detected_name = "Unknown"
                max_sim = 0
                if len(known_feats) > 0:
                    for k_idx, k_feat in enumerate(known_feats):
                        sim = compute_sim(embedding, k_feat)
                        if sim > max_sim:
                            max_sim = sim
                            if max_sim > REC_THRESHOLD:
                                detected_name = known_names[k_idx]

                # 2. 顔トラッキング (簡易距離マッチング)
                cx, cy = (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2
                matched_id = None
                min_dist = 200
                for pid, p in face_people_states.items():
                    if current_time - p.last_seen > 1.0: continue
                    pcx, pcy = (p.bbox[0]+p.bbox[2])/2, (p.bbox[1]+p.bbox[3])/2
                    dist = np.linalg.norm([cx-pcx, cy-pcy])
                    if dist < min_dist:
                        min_dist = dist
                        matched_id = pid
                
                if matched_id is None:
                    matched_id = face_next_id
                    face_people_states[matched_id] = FacePersonState(matched_id)
                    face_next_id += 1
                
                person = face_people_states[matched_id]
                person.last_seen = current_time
                person.bbox = bbox
                if detected_name != "Unknown": person.name = detected_name

                # 3. MediaPipe 居眠り詳細解析
                bx1, by1, bx2, by2 = bbox
                # 顔領域を少し広げてクロップ
                margin = int((bx2-bx1)*0.2)
                cx1, cy1 = max(0, bx1-margin), max(0, by1-margin)
                cx2, cy2 = min(w, bx2+margin), min(h, by2+margin)
                face_crop = frame[cy1:cy2, cx1:cx2]
                
                if face_crop.size > 0:
                    rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    mesh_res = face_mesh.process(rgb_crop)
                    
                    if mesh_res.multi_face_landmarks:
                        lms = mesh_res.multi_face_landmarks[0]
                        ch, cw, _ = face_crop.shape
                        pts = np.array([[int(lm.x*cw)+cx1, int(lm.y*ch)+cy1] for lm in lms.landmark])
                        
                        raw_ear = (calc_ear(pts[IDX_L_EYE]) + calc_ear(pts[IDX_R_EYE])) / 2.0
                        raw_mar = calc_mar(pts[IDX_MOUTH])
                        person.ear_buffer.append(raw_ear)
                        person.mar_buffer.append(raw_mar)
                        person.nose_y_buffer.append(pts[IDX_NOSE][1])

                        # ステータス判定
                        if len(person.ear_buffer) == SMOOTHING_WINDOW:
                            avg_ear = sum(person.ear_buffer)/SMOOTHING_WINDOW
                            
                            is_nodding = False
                            if len(person.nose_y_buffer) == NOD_FRAME_WINDOW:
                                y_diff = max(person.nose_y_buffer) - min(person.nose_y_buffer)
                                face_h = abs(pts[10][1] - pts[152][1])
                                if y_diff > (face_h * NOD_THRESHOLD):
                                    is_nodding = True
                            
                            person.status = "normal"
                            person.msg = ""
                            if avg_ear < THRESHOLD_EAR_CLOSE:
                                if not person.sleep_start: person.sleep_start = current_time
                                if current_time - person.sleep_start > TIME_SLEEP:
                                    person.status = "danger"
                                    person.msg = "SLEEP!"
                            else:
                                person.sleep_start = None
                                if avg_ear < THRESHOLD_EAR_HALF or is_nodding:
                                    if not person.utouto_start: person.utouto_start = current_time
                                    if current_time - person.utouto_start > TIME_UTOUTO:
                                        person.status = "warning"
                                        person.msg = "DROWSY"
                                else:
                                    person.utouto_start = None
                        
                        # 描画 (顔枠 & 情報)
                        f_color = (0,255,0)
                        if person.status == "danger": f_color = (0,0,255)
                        elif person.status == "warning": f_color = (0,165,255)
                        
                        # 顔枠
                        cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), f_color, 2)
                        # 名前と状態
                        info_text = f"{person.name} {person.msg}"
                        cv2.putText(annotated_frame, info_text, (bx1, by2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, f_color, 2)
                        
                        # 目のランドマーク (オプション)
                        for idx in IDX_L_EYE + IDX_R_EYE:
                            cv2.circle(annotated_frame, tuple(pts[idx]), 1, (255, 255, 0), -1)

            # ---------------------------------------------------------
            # [C] 記録 & 表示
            # ---------------------------------------------------------
            if trigger_alert_save:
                if current_time - last_alert_save > config.ALERT_INTERVAL:
                    save_executor.submit(save_snapshot_task, annotated_frame.copy(), "alert")
                    last_alert_save = current_time
            else:
                if current_time - last_normal_save > config.NORMAL_INTERVAL:
                    save_executor.submit(save_snapshot_task, annotated_frame.copy(), "normal")
                    last_normal_save = current_time

            # メモリ掃除
            pose_garbage = [i for i,t in pose_last_seen.items() if current_time - t > 60]
            for i in pose_garbage: del pose_people_states[i]
            face_garbage = [i for i,p in face_people_states.items() if current_time - p.last_seen > 5]
            for i in face_garbage: del face_people_states[i]

            # 画面表示
            cv2.imshow('Integrated AI Monitor (Face & Body)', annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception:
        traceback.print_exc()
    finally:
        save_executor.shutdown(wait=False)
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()