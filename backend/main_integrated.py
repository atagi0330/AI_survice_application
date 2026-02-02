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
THRESHOLD_EAR_CLOSE = 0.15
THRESHOLD_EAR_HALF  = 0.20
TIME_SLEEP   = 0.5  # テスト用に短縮（本来は2.0）
TIME_UTOUTO  = 0.3  # テスト用に短縮（本来は1.0）
NOD_FRAME_WINDOW = 30 
NOD_THRESHOLD    = 0.15 
THRESHOLD_MAR = 0.45      
TIME_LIMIT_YAWN = 0.5  # テスト用に短縮（本来は1.5）
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

def draw_info(img, bbox, name, msg, ear, mar, status):
    """顔情報を描画（face_only版と同じ）"""
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
    (tw, th), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(img, (x1, y2 + 5), (x1 + tw + 10, y2 + 5 + th + 10), (0, 0, 0), -1)
    
    tc = (255, 255, 255)
    if ear < THRESHOLD_EAR_CLOSE: tc = (0, 0, 255)
    elif ear < THRESHOLD_EAR_HALF: tc = (0, 255, 255)
    cv2.putText(img, info, (x1 + 5, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tc, 1)

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

def save_snapshot_task(frame, mode, filename=None):
    try:
        save_dir = os.path.join(config.LOG_DIR, mode)
        os.makedirs(save_dir, exist_ok=True)
        if filename is None:
            filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3] + ".jpg"
        
        cv2.imwrite(os.path.join(save_dir, filename), frame)

        # ★ 修正: モードに関係なく、50枚を超えたら古いものを消す
        files = sorted(glob.glob(os.path.join(save_dir, "*.jpg")))
        MAX_FILES = 50 
        
        # 50枚を超える分だけ、古い順に削除する
        while len(files) > MAX_FILES:
            try: 
                os.remove(files[0]) # 一番古いファイルを削除
                files.pop(0)        # リストからも削除
            except: 
                break
    except: pass
# ==========================================
# メイン処理
# ==========================================
def main():
    print("=== AI統合システム (異常時のみ記録版) 起動 ===")

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
# 3. YOLO Pose 初期化
    print(f"Initialize YOLO Pose ({config.POSE_MODEL_PATH})...")
    model_path = config.POSE_MODEL_PATH
    if not os.path.exists(model_path):
        model_path = "yolov26m-pose.pt" # Fallback

    # --- 修正ポイント：読み込み時に device を明示的に渡す ---
    yolo_model = YOLO(model_path, task='pose') 
    yolo_model.to(config.DEVICE) # PyTorchモデルとしてGPU転送を確実にする

    # 4. カメラ初期化
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
    if not cap.isOpened():
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("エラー: カメラが見つかりません。接続を確認してください。")
        return

    # 画質設定 (640x480で動作安定化)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # 状態管理
    pose_people_states = {} 
    pose_last_seen = {}
    
    face_people_states = {}
    face_next_id = 0

    last_alert_save = 0
    last_person_update = 0  # 人物情報の送信タイマー

    # ウィンドウ設定（800x600のウィンドウで表示）
    window_name = 'Integrated AI Monitor'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    print("=== システム稼働開始 (終了は 'q' キー) ===")

    try:
        while True:
            ret, frame = cap.read()
            
            # カメラ再接続ロジック
            if not ret:
                print("⚠️ 映像信号ロスト。再接続を試みます...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                continue
            
            current_time = time.time()
            h, w, _ = frame.shape
            annotated_frame = frame.copy()
            trigger_alert_save = False
            bx1, by1, bx2, by2 = 0, 0, 0, 0

            # ---------------------------------------------------------
            # [Step 1] 顔認識
            # ---------------------------------------------------------
            faces = app.get(frame)
            detected_faces_info = []

            for face in faces:
                name = "Unknown"
                if len(known_feats) > 0:
                    max_sim = 0
                    for k_idx, k_feat in enumerate(known_feats):
                        sim = compute_sim(face.embedding, k_feat)
                        if sim > max_sim:
                            max_sim = sim
                            if max_sim > REC_THRESHOLD:
                                name = known_names[k_idx]
                
                bbox = face.bbox.astype(int)
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                detected_faces_info.append({
                    "name": name,
                    "center": (cx, cy),
                    "bbox": bbox
                })

            # ---------------------------------------------------------
            # [Step 2] YOLO Pose (転倒・ふらつき)
            # ---------------------------------------------------------
            results = yolo_model.track(frame, persist=True, verbose=False, device=config.DEVICE, classes=[0])

            if results[0].boxes.id is not None:
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                keypoints_all = results[0].keypoints.data.cpu().numpy()
                boxes_all = results[0].boxes.xyxy.cpu().numpy()
                # -------------------------------------------------------------
                # ★ ここにID引き継ぎロジックを入れる
                # -------------------------------------------------------------
                # 1. 今回検出されたIDのリスト
                current_frame_ids = list(ids)

                # 2. 直前までいたけど、今回消えたIDを探す
                lost_ids = []
                for pid, last_t in pose_last_seen.items():
                   # 1秒以内に見失った、かつ 今回のリストにいない
                   if (current_time - last_t < 1.0) and (pid not in current_frame_ids):
                       lost_ids.append(pid)

                # 3. 今回新しく現れたIDを探す (まだ pose_people_states に登録されていないID)
                new_ids = [tid for tid in current_frame_ids if tid not in pose_people_states]

                # 4. マッチング処理
                for new_id in new_ids:
                    # 新しいIDの座標を取得
                    idx = list(ids).index(new_id)
                    box = boxes_all[idx] 
                    nx1, ny1, nx2, ny2 = box
                    ncx, ncy = (nx1+nx2)/2, (ny1+ny2)/2 # 新しい人の中心

                    best_match_old_id = None
                    min_dist = 100 # 近距離の閾値(ピクセル)

                    for old_id in lost_ids:
                        # 消えた人の最後の座標と比較
                        old_state = pose_people_states[old_id]
                        # bbox情報を持たせている前提 (持っていなければ保存が必要)
                        if hasattr(old_state, 'bbox'):
                             ox1, oy1, ox2, oy2 = old_state.bbox
                             ocx, ocy = (ox1+ox2)/2, (oy1+oy2)/2
                             
                             dist = np.linalg.norm([ncx - ocx, ncy - ocy])
                             if dist < min_dist:
                                 min_dist = dist
                                 best_match_old_id = old_id
                    
                    # 引き継ぎ実行
                    if best_match_old_id is not None:
                         # 新しいIDのエントリを作るときに、古いIDの中身（名前など）をコピー
                         print(f"ID引き継ぎ: {best_match_old_id} -> {new_id}")
                         # 古い情報をコピーして新しいIDに登録
                         pose_people_states[new_id] = pose_people_states[best_match_old_id]
                         # ただし履歴バッファなどはリセットするかそのままか考慮が必要
                         
                         # 古いIDは削除リストに入れるなりして処理完了
                         lost_ids.remove(best_match_old_id)

                # -------------------------------------------------------------
                for i, track_id in enumerate(ids):
                    if track_id not in pose_people_states:
                        pose_people_states[track_id] = features.PersonState()
                    
                    pose_last_seen[track_id] = current_time
                    state = pose_people_states[track_id]
                    kpts = keypoints_all[i]
                    box = boxes_all[i]
                    x1, y1, x2, y2 = box.astype(int)
                    
                    # bbox情報を追加（スナップショット切り抜き用）
                    state.bbox = [x1, y1, x2, y2]

                    # 名前紐付け
                    matched_name = ""
                    for face_info in detected_faces_info:
                        fcx, fcy = face_info["center"]
                        if x1 < fcx < x2 and y1 < fcy < y2:
                            matched_name = face_info["name"]
                            break

                    # 判定ロジック
                    is_pose_fall = features.check_fall_pose(kpts, box)
                    is_bad_posture_now = features.check_bad_posture(kpts)
                    is_stagger_now = features.check_staggering(kpts, state)

                    state.update_status(is_pose_fall, is_stagger_now, is_bad_posture_now)
                    if state.alert_active: trigger_alert_save = True

                    # 描画
                    color = state.status_color
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    label_parts = [f"ID:{track_id}"]
                    if matched_name and matched_name != "Unknown":
                        label_parts.append(matched_name)
                    label_parts.append(state.action_message)
                    label = " ".join(label_parts)

                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(annotated_frame, (x1, y1 - 25), (x1 + tw, y1), color, -1)
                    cv2.putText(annotated_frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # ---------------------------------------------------------
            # [Step 3] 居眠り検知
            # ---------------------------------------------------------
            for face_info in detected_faces_info:
                name = face_info["name"]
                bbox = face_info["bbox"]
                cx, cy = face_info["center"]
                
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
                if name != "Unknown": person.name = name

                # MediaPipe解析
                bx1, by1, bx2, by2 = bbox
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

                        if len(person.ear_buffer) == SMOOTHING_WINDOW:
                            avg_ear = sum(person.ear_buffer)/SMOOTHING_WINDOW
                            avg_mar = sum(person.mar_buffer)/SMOOTHING_WINDOW
                            
                            # face_only版と同じ形式
                            face_h_len = np.linalg.norm(pts[10] - pts[152])
                            
                            person.status = "normal"
                            person.msg = ""
                            
                            is_nodding = False
                            if len(person.nose_y_buffer) == NOD_FRAME_WINDOW:
                                y_range = max(person.nose_y_buffer) - min(person.nose_y_buffer)
                                if y_range > (face_h_len * NOD_THRESHOLD): is_nodding = True
                            
                            # 居眠り検知（face_only版と同じ）
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
                            
                            # アラートがあれば保存フラグを立てる
                            if person.status != "normal":
                                trigger_alert_save = True

                            # face_only版と同じdraw_info関数で描画
                            draw_info(annotated_frame, bbox, person.name, person.msg, avg_ear, avg_mar, person.status)

            # アラート保存用フラグ (毎フレーム初期化)
            trigger_alert_save = False
            # 変数初期化 (NameError防止)
            bx1, by1, bx2, by2 = 0, 0, 0, 0

            # -------------------------------------------------------------
            # [Step 4] 優先順位付きアラート判定 & 記録
            # -------------------------------------------------------------
            # 人物情報をサーバーに定期送信（2秒ごと）
            if current_time - last_person_update > 2.0:
                last_person_update = current_time
                try:
                    import requests
                    # 検出された全人物の情報を収集
                    persons_data = []

                    # 優先順位判定用変数
                    best_alert_person = None
                    best_alert_score = -1 
                    matched_body_ids = set() # 顔と紐付いた体ID

                    def get_alert_score(msg_str):
                        m = msg_str.upper()
                        if "FALL" in m or "転倒" in m: return 100
                        if "STAGGER" in m or "ふらつき" in m: return 90
                        if "SLEEP" in m or "居眠り" in m: return 50
                        if "DROWSY" in m or "NOD" in m: return 40
                        if "YAWN" in m or "あくび" in m: return 30
                        if "NORMAL" in m or "正常" in m or m == "": return 0
                        return 10 # その他異常

                    # 1. 顔検知ループ (Face -> Body Match)
                    for face_info in detected_faces_info:
                        name = face_info["name"]
                        if name != "Unknown":
                            # face_people_statesから状態を取得 (Status: 顔起因)
                            status = "正常"
                            face_bbox = face_info["bbox"]
                            face_cx = (face_bbox[0] + face_bbox[2]) / 2
                            face_cy = (face_bbox[1] + face_bbox[3]) / 2

                            for pid, p in face_people_states.items():
                                if p.name == name:
                                    if p.status != "normal":
                                        status = p.msg 
                                    break
                            
                            # pose_people_statesから状態を取得 (Emergency Status: 体起因)
                            emergency_status = "無"
                            for pid, p in pose_people_states.items():
                                if not hasattr(p, 'bbox'): continue
                                pb = p.bbox
                                # 顔中心が体枠内にあるかチェック
                                if (pb[0] < face_cx < pb[2]) and (pb[1] < face_cy < pb[3]):
                                    matched_body_ids.add(pid) # マッチしたIDを記録
                                    if p.msg != "Normal":
                                        emergency_status = p.msg
                                    break

                            # 異常時のみフロントエンドに反映
                            if status != "正常" or emergency_status != "無":
                                persons_data.append({
                                    "name": name,
                                    "status": status,
                                    "emergency_status": emergency_status
                                })
                                
                                # アラート候補として評価
                                msg_cand = emergency_status if emergency_status != "無" else status
                                score = get_alert_score(msg_cand)

                                # 現在のベストスコアより高ければ採用
                                if score > best_alert_score and score > 0:
                                    best_alert_score = score
                                    
                                    # ダミーコンテナ作成
                                    class AlertPerson: pass
                                    ap = AlertPerson()
                                    ap.name = name
                                    ap.msg = msg_cand
                                    ap.status = "danger" if score >= 90 else "warning"
                                    
                                    best_alert_person = ap
                                    # bbox情報 (顔)
                                    bx1, by1, bx2, by2 = map(int, face_info["bbox"])
                    
                    # 2. 体のみ検知ループ (Body Only)
                    # 顔とマッチしなかった、かつ異常があるものをチェック
                    for pid, p in pose_people_states.items():
                        if pid in matched_body_ids: continue # 既に処理済み
                        
                        if p.msg != "Normal":
                            # アラート候補として評価
                            score = get_alert_score(p.msg)
                            
                            if score > best_alert_score and score > 0:
                                best_alert_score = score
                                
                                class AlertPerson: pass
                                ap = AlertPerson()
                                ap.name = "Unknown"
                                ap.msg = p.msg
                                ap.status = p.status # features.pyで設定済み
                                
                                best_alert_person = ap
                                if hasattr(p, 'bbox'):
                                    bx1, by1, bx2, by2 = map(int, p.bbox)
                                else:
                                    bx1, by1, bx2, by2 = 0, 0, 100, 100

                    # 3. ベストなアラートがあればトリガー
                    if best_alert_person is not None:
                        trigger_alert_save = True
                        person = best_alert_person

                    if persons_data:
                        save_executor.submit(send_api_request, "http://localhost:8000/update/persons", {
                            "persons": persons_data
                        })
                except: pass

            # 異常時のみスナップショット保存
            if trigger_alert_save:
                if current_time - last_alert_save > config.ALERT_INTERVAL:
                    # ファイル名をここで生成
                    filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3] + ".jpg"
                    
                    # 修正: 体全体（YOLO検知枠）を切り出し
                    # デフォルトは顔枠（見つからなかった場合用）
                    crop_x1, crop_y1, crop_x2, crop_y2 = bx1, by1, bx2, by2
                    
                    # 顔の中心点
                    face_cx = (bx1 + bx2) / 2
                    face_cy = (by1 + by2) / 2
                    
                    # YOLOの人物検知結果から、この顔を含む体を探す
                    for pid, p in pose_people_states.items():
                        if not hasattr(p, 'bbox'): continue
                        
                        # YOLOのbboxは [x1, y1, x2, y2]
                        pb = p.bbox
                        if (pb[0] < face_cx < pb[2]) and (pb[1] < face_cy < pb[3]):
                            # 包含関係にある体が見つかった場合、その枠を採用
                            crop_x1 = int(pb[0])
                            crop_y1 = int(pb[1])
                            crop_x2 = int(pb[2])
                            crop_y2 = int(pb[3])
                            break

                    h, w, _ = frame.shape
                    # 範囲チェック
                    crop_x1 = max(0, crop_x1)
                    crop_y1 = max(0, crop_y1)
                    crop_x2 = min(w, crop_x2)
                    crop_y2 = min(h, crop_y2)
                    
                    # 修正: 描画が入る前の 'frame' から切り抜く（緑枠などを消すため）
                    face_crop_img = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()

                    # 修正: 画像が大きすぎるためリサイズ (横幅150px基準)
                    try:
                        h_crop, w_crop = face_crop_img.shape[:2]
                        if w_crop > 350:
                            aspect_ratio = h_crop / w_crop
                            new_w = 350
                            new_h = int(new_w * aspect_ratio)
                            face_crop_img = cv2.resize(face_crop_img, (new_w, new_h))
                    except: pass

                    # 保存タスクにファイル名を渡す
                    save_executor.submit(save_snapshot_task, face_crop_img, "alert", filename)
                    last_alert_save = current_time
                    
                    try:
                        # msgの内容からkindを判定
                        msg_upper = person.msg.upper()
                        kind = "WARN"
                        if "FALL" in msg_upper: kind = "FALL"
                        elif "STAGGER" in msg_upper: kind = "STAGGER"
                        elif "SLEEP" in msg_upper: kind = "SLEEP"
                        elif "DROWSY" in msg_upper: kind = "DROWSY"
                        elif "YAWN" in msg_upper: kind = "YAWN"
                        elif "POSTURE" in msg_upper: kind = "POSTURE"
                        
                        print(f"DEBUG SEND: Name={person.name}, Msg=[{person.msg}], Kind=[{kind}]")
                        save_executor.submit(send_api_request, "http://localhost:8000/update/event", {
                            "name": person.name,
                            "msg": person.msg,
                            "kind": kind,
                            "snapshot": f"alert/{filename}"
                        })
                    except: pass

            # メモリ掃除
            pose_garbage = [i for i, t in pose_last_seen.items() if current_time - t > 60]
            for i in pose_garbage:
                if i in pose_people_states: del pose_people_states[i]
                del pose_last_seen[i]

            face_garbage = [i for i, p in face_people_states.items() if current_time - p.last_seen > 5]
            for i in face_garbage:
                if i in face_people_states: del face_people_states[i]

            cv2.imshow(window_name, annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception:
        traceback.print_exc()
    finally:
        save_executor.shutdown(wait=False)
        if 'cap' in locals() and cap.isOpened(): cap.release()
        cv2.destroyAllWindows()

def send_api_request(url, json_data):
    """APIリクエストを非同期で送信するためのヘルパー関数"""
    try:
        import requests
        requests.post(url, json=json_data, timeout=1)
    except Exception:
        pass # エラーは無視（メインループを止めないため）
if __name__ == "__main__":
    main()