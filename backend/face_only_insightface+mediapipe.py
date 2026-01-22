import os
import sys

# ログ抑制
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
import time
import mediapipe as mp
import insightface
from insightface.app import FaceAnalysis
from collections import deque
import warnings
import glob

# 警告非表示
warnings.simplefilter("ignore")

# ==========================================
# ★ 設定
# ==========================================
DET_SIZE = (1280, 1280) # 検出サイズを少し下げて汎用性を上げる
DET_THRESH = 0.5
REC_THRESHOLD = 0.35  # 顔認証の判定閾値

# --- 目の判定 (EAR) ---
THRESHOLD_EAR_CLOSE = 0.13
THRESHOLD_EAR_HALF  = 0.20
TIME_SLEEP   = 2.0
TIME_UTOUTO  = 1.0

# --- その他判定 ---
NOD_FRAME_WINDOW = 30 
NOD_THRESHOLD    = 0.15 
THRESHOLD_MAR = 0.45      
TIME_LIMIT_YAWN = 1.5   
SMOOTHING_WINDOW = 6

# ==========================================
# MediaPipe ランドマーク定義
# ==========================================
IDX_L_EYE = [33, 160, 158, 133, 153, 144]
IDX_R_EYE = [362, 385, 387, 263, 373, 380]
IDX_MOUTH = [13, 14, 61, 291]
IDX_NOSE  = 1 

# ==========================================
# 計算ロジック
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

def draw_info(img, bbox, name, msg, ear, mar, status):
    x1, y1, x2, y2 = bbox
    color = (0, 255, 0)
    if status == "danger": color = (0, 0, 255)
    elif status == "warning": color = (0, 165, 255)
    elif status == "caution": color = (0, 255, 255)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    name_disp = f"{name}" if name else "Unknown"
    cv2.putText(img, name_disp, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(img, msg, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    info = f"EAR:{ear:.2f} MAR:{mar:.2f}"
    (w, h), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(img, (x1, y2 + 5), (x1 + w + 10, y2 + 5 + h + 10), (0, 0, 0), -1)
    
    tc = (255, 255, 255)
    if ear < THRESHOLD_EAR_CLOSE: tc = (0, 0, 255)
    elif ear < THRESHOLD_EAR_HALF: tc = (0, 255, 255)
    cv2.putText(img, info, (x1 + 5, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tc, 1)

# ==========================================
# クラス
# ==========================================
class PersonState:
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
# ★ 変更点: 日本語パス対応 & 自動回転機能
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(CURRENT_DIR, "face_db")

def imread_safe(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(f"Read Error: {e}")
        return None

def load_known_faces(app, directory=DEFAULT_DB_PATH):
    known_features = []
    known_names = []
    
    print("\n" + "="*50)
    print("【顔データベース読み込み開始 (自動回転対応版)】")
    print(f" 📂 フォルダ: {directory}")
    print("="*50)

    if not os.path.exists(directory):
        print(f" ❌ エラー: フォルダが見つかりません！")
        print(f"    -> 自動作成します: {directory}")
        os.makedirs(directory, exist_ok=True)
        return [], []

    files = glob.glob(os.path.join(directory, '*'))
    valid_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    if len(valid_files) == 0:
        print(" ⚠️  警告: 画像ファイルがありません！")
    
    for file_path in valid_files:
        basename = os.path.basename(file_path)
        img_org = imread_safe(file_path)
        
        if img_org is None:
            print(f" ❌ 読込不可: {basename}")
            continue
        
        # --- 自動回転ロジック ---
        # 0度, 90度, 180度, 270度 と回転させて顔が見つかるまで試す
        found_face = False
        img_temp = img_org.copy()
        
        for angle in [0, 90, 180, 270]:
            faces = app.get(img_temp)
            
            if len(faces) > 0:
                # 顔が見つかった！
                faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
                known_features.append(faces[0].embedding)
                name, _ = os.path.splitext(basename)
                known_names.append(name)
                
                msg = "成功"
                if angle > 0: msg += f" (画像回転補正: {angle}度)"
                print(f" ✅ {msg}: {name} (ファイル: {basename})")
                found_face = True
                break
            else:
                # 顔が見つからない場合、画像を90度回転させて次へ
                img_temp = cv2.rotate(img_temp, cv2.ROTATE_90_CLOCKWISE)
        
        if not found_face:
            print(f" ❌ 顔検出失敗: {basename}")
            print(f"    -> 画像が暗すぎるか、顔がアップすぎる、または小さすぎる可能性があります。")
            # どんな画像を読み込んだか確認用に出力（デバッグ用）
            debug_path = os.path.join(CURRENT_DIR, "debug_failed_" + basename)
            cv2.imwrite(debug_path, img_org)
            print(f"    -> 確認用に画像を保存しました: {debug_path}")

    print("-" * 50)
    print(f"【結果】 合計 {len(known_names)} 人の顔を登録しました。")
    print("=" * 50 + "\n")
            
    return known_features, known_names

def main():
    print("★ システム初期化中...")

    # 1. InsightFace
    # ★ DET_SIZEを自動にして検出力を上げる
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640)) 

    # 読み込み実行
    known_feats, known_names = load_known_faces(app)

    # 2. MediaPipe
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.3,
        static_image_mode=False
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    people_states = {} 
    next_id = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        h, w, _ = frame.shape

        # ==================================================
        # ★ 画面左上にステータス表示
        # ==================================================
        cv2.rectangle(frame, (0, 0), (400, 40), (0, 0, 0), -1) 
        status_text = f"Registered Faces: {len(known_names)}"
        cv2.putText(frame, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if len(known_names) == 0:
             cv2.putText(frame, "WARNING: NO DB LOADED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # --------------------------------------------------

        # A. InsightFace
        faces = app.get(frame)
        
        for face in faces:
            bbox = face.bbox.astype(int)
            embedding = face.embedding

            # --- 顔認証 ---
            detected_name = "Unknown"
            max_sim = 0
            if len(known_feats) > 0:
                best_idx = -1
                for i, k_feat in enumerate(known_feats):
                    sim = compute_sim(embedding, k_feat)
                    if sim > max_sim:
                        max_sim = sim
                        best_idx = i
                
                if max_sim > REC_THRESHOLD:
                    detected_name = known_names[best_idx]
                
                if best_idx != -1:
                    print(f"\r検出: {known_names[best_idx]} (スコア: {max_sim:.4f}) -> 判定: {detected_name}", end="")

            # --- トラッキング ---
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            matched_id = None
            min_dist = 200 
            
            for pid, p_state in people_states.items():
                if time.time() - p_state.last_seen > 1.0: continue
                p_cx = (p_state.bbox[0] + p_state.bbox[2]) / 2
                p_cy = (p_state.bbox[1] + p_state.bbox[3]) / 2
                dist = np.linalg.norm([center_x - p_cx, center_y - p_cy])
                if dist < min_dist:
                    min_dist = dist
                    matched_id = pid
            
            if matched_id is None:
                matched_id = next_id
                people_states[matched_id] = PersonState(matched_id)
                next_id += 1
            
            person = people_states[matched_id]
            person.last_seen = time.time()
            person.bbox = bbox
            
            if detected_name != "Unknown":
                person.name = detected_name

            # B. MediaPipe
            x1, y1, x2, y2 = bbox
            margin_x = int((x2 - x1) * 0.2)
            margin_y = int((y2 - y1) * 0.2)
            cx1, cy1 = max(0, x1 - margin_x), max(0, y1 - margin_y)
            cx2, cy2 = min(w, x2 + margin_x), min(h, y2 + margin_y)
            
            face_crop = frame[cy1:cy2, cx1:cx2]
            if face_crop.size == 0: continue

            rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            mesh_result = face_mesh.process(rgb_crop)
            
            if mesh_result.multi_face_landmarks:
                lms = mesh_result.multi_face_landmarks[0]
                ch, cw, _ = face_crop.shape
                
                pts = []
                for lm in lms.landmark:
                    pts.append([int(lm.x * cw) + cx1, int(lm.y * ch) + cy1])
                pts = np.array(pts)

                face_h_len = np.linalg.norm(pts[10] - pts[152])
                nose_y = pts[IDX_NOSE][1]
                
                raw_ear = (calc_ear(pts[IDX_L_EYE]) + calc_ear(pts[IDX_R_EYE])) / 2.0
                raw_mar = calc_mar(pts[IDX_MOUTH])
                
                person.ear_buffer.append(raw_ear)
                person.mar_buffer.append(raw_mar)
                person.nose_y_buffer.append(nose_y)
                
                if len(person.ear_buffer) == SMOOTHING_WINDOW:
                    avg_ear = sum(person.ear_buffer) / SMOOTHING_WINDOW
                    avg_mar = sum(person.mar_buffer) / SMOOTHING_WINDOW
                    
                    person.status = "normal"
                    person.msg = "" 

                    is_nodding = False
                    if len(person.nose_y_buffer) == NOD_FRAME_WINDOW:
                        y_range = max(person.nose_y_buffer) - min(person.nose_y_buffer)
                        if y_range > (face_h_len * NOD_THRESHOLD): is_nodding = True

                    if avg_ear < THRESHOLD_EAR_CLOSE:
                        if person.sleep_start is None: person.sleep_start = time.time()
                        if time.time() - person.sleep_start > TIME_SLEEP:
                            person.status = "danger"
                            person.msg = "SLEEP!"
                    else:
                        person.sleep_start = None
                        if (avg_ear < THRESHOLD_EAR_HALF) or is_nodding:
                            if person.utouto_start is None: person.utouto_start = time.time()
                            if time.time() - person.utouto_start > TIME_UTOUTO:
                                person.status = "warning"
                                person.msg = "DROWSY"
                                if is_nodding: person.msg += "(Nod)"
                        else:
                            person.utouto_start = None

                        if person.status == "normal":
                            if avg_mar > THRESHOLD_MAR:
                                if person.yawn_start is None: person.yawn_start = time.time()
                                if time.time() - person.yawn_start > TIME_LIMIT_YAWN:
                                    person.status = "caution"
                                    person.msg = "YAWN"
                            else:
                                person.yawn_start = None

                    draw_info(frame, bbox, person.name, person.msg, avg_ear, avg_mar, person.status)
                    
                    color = (0, 255, 0)
                    if person.status == "danger": color = (0, 0, 255)
                    elif person.status == "warning": color = (0, 165, 255)
                    for idx in IDX_L_EYE + IDX_R_EYE + IDX_MOUTH:
                        cv2.circle(frame, tuple(pts[idx]), 1, color, -1)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 2)

        garbage = [pid for pid, p in people_states.items() if time.time() - p.last_seen > 5.0]
        for pid in garbage: del people_states[pid]

        cv2.imshow('Ultimate Hybrid System (Recognition)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()