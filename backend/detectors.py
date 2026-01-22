# detectors.py
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import config

class BodyDetector:
    """YOLOv8-Pose を使った全身検出クラス"""
    def __init__(self, model_size='n'):
        print(f"Loading YOLOv8{model_size}-pose...")
        self.model = YOLO(f'yolov8{model_size}-pose.pt')
        self.model.to(config.DEVICE)

    def detect(self, frame):
        """フレームから人を検出して結果を返す"""
        # classes=[0] で人のみ検出
        results = self.model.track(frame, persist=True, verbose=False, device=config.DEVICE, classes=[0])
        
        parsed_results = []
        if results[0].boxes.id is not None:
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            kpts = results[0].keypoints.data.cpu().numpy() # [x, y, conf]
            boxes = results[0].boxes.xyxy.cpu().numpy()
            
            for i, tid in enumerate(ids):
                parsed_results.append({
                    "id": tid,
                    "box": boxes[i].astype(int),
                    "keypoints": kpts[i]
                })
        return parsed_results

class FaceDetector:
    """MediaPipe Face Mesh を使った顔詳細分析クラス"""
    def __init__(self):
        print("Loading MediaPipe FaceMesh...")
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            max_num_faces=1, # 切り抜き画像には1つの顔しかないはず
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        # インデックス定義
        self.IDX_L_EYE = [33, 160, 158, 133, 153, 144]
        self.IDX_R_EYE = [362, 385, 387, 263, 373, 380]
        self.IDX_MOUTH = [13, 312, 317, 14, 87, 82]
        self.IDX_CHEEK = [123, 50, 205]

    def analyze(self, frame, face_crop):
        """
        切り抜かれた顔画像(face_crop)を分析する
        frameは顔色取得のために元の画像データが必要な場合に使用
        """
        if face_crop.size == 0: return None

        rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_crop)

        if not results.multi_face_landmarks:
            return None

        # 検出されたランドマーク
        lms = results.multi_face_landmarks[0]
        h, w, _ = face_crop.shape
        
        # 座標変換 (正規化座標 -> ピクセル座標)
        pts = np.array([[int(lm.x * w), int(lm.y * h)] for lm in lms.landmark])

        # EAR (目)
        left_ear = self._calc_ear(pts[self.IDX_L_EYE])
        right_ear = self._calc_ear(pts[self.IDX_R_EYE])
        avg_ear = (left_ear + right_ear) / 2.0
        is_sleep = avg_ear < config.THRESHOLD_EAR

        # MAR (あくび)
        mar = self._calc_mar(pts[self.IDX_MOUTH])
        is_yawn = mar > config.THRESHOLD_MAR

        # 顔色 (H値)
        face_color_h = self._get_color(face_crop, pts[self.IDX_CHEEK])

        return {
            "is_sleeping": is_sleep,
            "is_yawning": is_yawn,
            "face_color": face_color_h,
            "ear": avg_ear # デバッグ用
        }

    def _calc_ear(self, pts):
        v1 = np.linalg.norm(pts[1] - pts[5])
        v2 = np.linalg.norm(pts[2] - pts[4])
        h = np.linalg.norm(pts[0] - pts[3])
        return (v1+v2)/(2.0*h) if h!=0 else 0

    def _calc_mar(self, pts):
        v = np.linalg.norm(pts[2] - pts[3])
        h = np.linalg.norm(pts[0] - pts[1])
        return v/h if h!=0 else 0

    def _get_color(self, img, pts):
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
        mean = cv2.mean(img, mask=mask)[:3]
        return cv2.cvtColor(np.uint8([[mean]]), cv2.COLOR_BGR2HSV)[0][0][0]