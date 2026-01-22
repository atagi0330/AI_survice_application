import cv2
import time
import traceback
import os
import glob
import datetime
import face_recognition
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO

import config
import features

# 画像保存ワーカー
save_executor = ThreadPoolExecutor(max_workers=1)

# ★ 顔DBのロード関数
def load_face_database(db_path="face_db"):
    known_encodings = []
    known_names = []
    
    if not os.path.exists(db_path):
        os.makedirs(db_path)
        print(f"警告: {db_path} フォルダがありません。作成しました。")
        return known_encodings, known_names

    print("顔データベースを読み込み中...")
    for file_path in glob.glob(os.path.join(db_path, "*.*")):
        try:
            # 日本語ファイル名対応
            img_bgr = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img_bgr is None: continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            encodings = face_recognition.face_encodings(img_rgb)
            if len(encodings) > 0:
                known_encodings.append(encodings[0])
                name = os.path.splitext(os.path.basename(file_path))[0]
                known_names.append(name)
                print(f"登録: {name}")
        except Exception as e:
            print(f"スキップ: {file_path} ({e})")
            
    return known_encodings, known_names

def save_snapshot_task(frame, mode):
    try:
        save_dir = os.path.join(config.LOG_DIR, mode)
        os.makedirs(save_dir, exist_ok=True)
        filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3] + ".jpg"
        cv2.imwrite(os.path.join(save_dir, filename), frame)
        if mode == "normal":
            files = sorted(glob.glob(os.path.join(save_dir, "*.jpg")))
            if len(files) > config.NORMAL_MAX_FILES:
                try: os.remove(files[0])
                except: pass
    except Exception as e:
        print(f"Save Error: {e}")

def main():
    print(f"★ 実行デバイス: {config.DEVICE}")

    # ★ 顔DBのロード
    known_face_encodings, known_face_names = load_face_database()

    try:
        # --- YOLOモデル読み込み ---
        model_path = config.POSE_MODEL_PATH
        print(f"YOLO Poseモデルをロード中... ({model_path})")

        if not os.path.exists(model_path):
            alt_path = os.path.join("AI_service", model_path)
            if os.path.exists(alt_path):
                print(f"指定モデルがカレントに無いため 'AI_service/{model_path}' を使用します。")
                model_path = alt_path
            else:
                fallback = "yolov8n-pose.pt" if os.path.exists("yolov8n-pose.pt") else "yolov8m-pose.pt"
                print(f"警告: 指定モデル '{config.POSE_MODEL_PATH}' が見つかりません。代わりに '{fallback}' を使用します。")
                model_path = fallback

        yolo_model = YOLO(model_path)
        yolo_model.to(config.DEVICE)

        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            print("エラー: カメラなし")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        people_states = {} 
        last_seen_times = {}
        id_name_map = {} 

        last_normal_save_time = time.time()
        last_alert_save_time = 0

        print("=== システム稼働開始 ===")

        while True:
            ret, frame = cap.read()
            if not ret: break
            
            current_time = time.time()
            annotated_frame = frame.copy()
            trigger_alert_save = False

            results = yolo_model.track(frame, persist=True, verbose=False, device=config.DEVICE, classes=[0])

            if results[0].boxes.id is not None:
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                keypoints_all = results[0].keypoints.data.cpu().numpy()
                boxes_all = results[0].boxes.xyxy.cpu().numpy()

                for i, track_id in enumerate(ids):
                    if track_id not in people_states:
                        people_states[track_id] = features.PersonState()
                    
                    last_seen_times[track_id] = current_time
                    state = people_states[track_id]
                    kpts = keypoints_all[i]
                    box = boxes_all[i]
                    
                    x1, y1, x2, y2 = box.astype(int)

                    # --- ★ 顔認証ロジック ---
                    display_name = id_name_map.get(track_id, "Unknown")
                    if display_name == "Unknown":
                        h, w, _ = frame.shape
                        bx1, by1 = max(0, x1), max(0, y1)
                        bx2, by2 = min(w, x2), min(h, y2)
                        person_img = frame[by1:by2, bx1:bx2]
                        
                        if person_img.shape[0] > 50 and person_img.shape[1] > 50:
                            rgb_person = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
                            face_locs = face_recognition.face_locations(rgb_person, model="hog")
                            if face_locs:
                                face_encs = face_recognition.face_encodings(rgb_person, face_locs)
                                if face_encs:
                                    matches = face_recognition.compare_faces(known_face_encodings, face_encs[0], tolerance=0.5)
                                    name = "Unknown"
                                    face_distances = face_recognition.face_distance(known_face_encodings, face_encs[0])
                                    if len(face_distances) > 0:
                                        best_match_index = np.argmin(face_distances)
                                        if matches[best_match_index]:
                                            name = known_face_names[best_match_index]
                                    if name != "Unknown":
                                        id_name_map[track_id] = name
                                        display_name = name

                    # --- 1. 現瞬間の検知 ---
                    is_pose_fall = features.check_fall_pose(kpts, box)
                    is_bad_posture_now = features.check_bad_posture(kpts)
                    # ★ ここで新しい高精度ふらつき判定が呼ばれます
                    is_stagger_now = features.check_staggering(kpts, state)

                    # --- 2. 時間経過チェック ---
                    state.update_status(is_pose_fall, is_stagger_now, is_bad_posture_now)

                    if state.alert_active:
                        trigger_alert_save = True

                    # --- 3. 描画 ---
                    color = state.status_color
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    msg = state.action_message
                    (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(annotated_frame, (x1, y1-30), (x1+tw, y1), color, -1)
                    cv2.putText(annotated_frame, msg, (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    info_text = f"ID:{track_id} {display_name}"
                    cv2.putText(annotated_frame, info_text, (x1, y1-40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # 画像保存
            if trigger_alert_save:
                if current_time - last_alert_save_time > config.ALERT_INTERVAL:
                    save_executor.submit(save_snapshot_task, annotated_frame.copy(), "alert")
                    last_alert_save_time = current_time
            else:
                if current_time - last_normal_save_time > config.NORMAL_INTERVAL:
                    save_executor.submit(save_snapshot_task, annotated_frame.copy(), "normal")
                    last_normal_save_time = current_time

            # メモリ掃除
            garbage_ids = [tid for tid, t in last_seen_times.items() if current_time - t > 60]
            for tid in garbage_ids:
                if tid in people_states: del people_states[tid]
                if tid in id_name_map: del id_name_map[tid]

            cv2.imshow('AI Health Management System (Face ID)', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    except Exception:
        print("\n" + "="*60)
        traceback.print_exc()
        print("="*60)
        input("エラー終了")

    finally:
        save_executor.shutdown(wait=False)
        if 'cap' in locals() and cap.isOpened(): cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()