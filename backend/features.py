# features.py
import time
import numpy as np
from collections import deque # ★追加
import config

class PersonState:
    def __init__(self):
        self.state = "Normal"
        self.start_time = time.time()
        self.status_color = (0, 255, 0)
        self.action_message = "Normal"
        self.alert_active = False

        # --- ★ ふらつき検知用の履歴バッファ (dequeに変更) ---
        # 体幹角度の履歴 (左右の揺れを見る)
        self.body_angle_hist = deque(maxlen=config.STAGGER_HISTORY_LEN)
        # 腰の中心X座標の履歴 (千鳥足を見る)
        self.center_x_hist = deque(maxlen=config.STAGGER_HISTORY_LEN)

        # 各状態の継続開始時間
        self.t_fall_start = None
        self.t_posture_start = None
        self.t_stagger_start = None

    def update_status(self, is_fall, is_stagger, is_bad_posture):
        now = time.time()
        
        # 1. 転倒 (最優先)
        if is_fall:
            if self.t_fall_start is None: self.t_fall_start = now
            if now - self.t_fall_start >= config.DURATION_LIMIT_FALL:
                self.state = "Fall"
                self.status_color = (0, 0, 255) # Red
                self.action_message = "WARNING: FALL DETECTED!"
                self.alert_active = True
                return
        else:
            self.t_fall_start = None

        # 2. ふらつき判定
        if is_stagger:
            if self.t_stagger_start is None: self.t_stagger_start = now
            if now - self.t_stagger_start >= config.DURATION_LIMIT_STAGGER:
                self.state = "Stagger"
                self.status_color = (0, 165, 255) # Orange
                self.action_message = "Alert: Please take a rest."
                self.alert_active = True
                return
        else:
            self.t_stagger_start = None

        # 3. 姿勢悪化判定
        if is_bad_posture:
            if self.t_posture_start is None: self.t_posture_start = now
            if now - self.t_posture_start >= config.DURATION_LIMIT_POSTURE:
                self.state = "BadPosture"
                self.status_color = (255, 100, 0) # Orange
                self.action_message = "Advice: Let's stretch!"
                self.alert_active = False 
                return
        else:
            self.t_posture_start = None

        # 4. 正常
        self.state = "Normal"
        self.status_color = (0, 255, 0)
        self.action_message = "Good Condition"
        self.alert_active = False


def get_point(kpts, idx):
    if kpts[idx][2] < config.KEYPOINT_CONF_TH:
        return None
    return kpts[idx][:2]

def check_fall_pose(kpts, bbox=None):
    # (省略: 以前と同じコードを使用してください)
    # ここには以前の check_fall_pose の中身が入ります
    # 簡易版として再掲します
    l_sh = get_point(kpts, 5)
    r_sh = get_point(kpts, 6)
    l_hip = get_point(kpts, 11)
    r_hip = get_point(kpts, 12)
    if l_sh is None or r_sh is None or l_hip is None or r_hip is None: return False

    if bbox is not None:
        _, y1, _, y2 = bbox
        h = float(max(1, y2 - y1))
        w = float(max(1, bbox[2] - bbox[0]))
    else:
        ys = [p[1] for p in [l_sh, r_sh, l_hip, r_hip]]
        xs = [p[0] for p in [l_sh, r_sh, l_hip, r_hip]]
        h = float(max(1, max(ys) - min(ys)))
        w = float(max(1, max(xs) - min(xs)))

    if h < config.FALL_MIN_HEIGHT: return False

    aspect = w / h
    mid_sh = (l_sh + r_sh) / 2
    mid_hip = (l_hip + r_hip) / 2
    dx = abs(mid_sh[0] - mid_hip[0])
    dy = abs(mid_sh[1] - mid_hip[1])

    if (dy / h < config.FALL_VERTICAL_RATIO) or (aspect > config.FALL_ASPECT_RATIO and dx > dy):
        return True
    return False

def check_bad_posture(kpts):
    # (省略: 以前と同じコードを使用してください)
    l_sh = get_point(kpts, 5)
    l_hip = get_point(kpts, 11)
    l_knee = get_point(kpts, 13)
    if l_sh is not None and l_hip is not None and l_knee is not None:
        thigh_dy = abs(l_hip[1] - l_knee[1])
        torso_dy = abs(l_sh[1] - l_hip[1])
        if thigh_dy > 0 and (torso_dy / thigh_dy < config.POSTURE_RATIO_LIMIT):
            return True
    return False

def check_staggering(kpts, state):
    """
    ★ 高精度ふらつき判定 (Zigzag Path & Body Sway)
    """
    # 必要な点: 左右の肩(5,6)と左右の腰(11,12)
    l_sh = get_point(kpts, 5)
    r_sh = get_point(kpts, 6)
    l_hip = get_point(kpts, 11)
    r_hip = get_point(kpts, 12)

    if (l_sh is None or r_sh is None or l_hip is None or r_hip is None):
        return False

    # 1. 特徴点計算
    mid_shoulder = (l_sh + r_sh) / 2
    mid_hip = (l_hip + r_hip) / 2
    
    # --- ロジックA: 体幹の傾き (Body Sway) ---
    dx = mid_shoulder[0] - mid_hip[0]
    dy = mid_shoulder[1] - mid_hip[1]
    
    # 垂直からの角度 (度数法)
    # 歩行中は多少前傾するが、左右(dx)にブレるのは異常
    if dy != 0:
        angle = np.degrees(np.arctan2(dx, -dy)) 
    else:
        angle = 0
    state.body_angle_hist.append(angle)

    # --- ロジックB: 重心のブレ (Zigzag Path) ---
    state.center_x_hist.append(mid_hip[0])

    # --- 判定 (データが溜まってから) ---
    if len(state.body_angle_hist) < config.STAGGER_HISTORY_LEN:
        return False

    # 評価1: 角度の「標準偏差」 (揺れの激しさ)
    angle_variance = np.std(state.body_angle_hist)
    
    # 評価2: 軌道の「滑らかさからの逸脱」 (千鳥足)
    x_history = list(state.center_x_hist)
    # 移動平均 (直近5フレームの平均) を作成して滑らかな軌道を作る
    smooth_x = np.convolve(x_history, np.ones(5)/5, mode='valid') 
    # 現在の軌道と比較
    current_x_segment = x_history[-len(smooth_x):]
    path_deviation = np.mean(np.abs(np.array(current_x_segment) - smooth_x))

    # ★ 判定条件 (どちらかがしきい値を超えたら「ふらつき」)
    if angle_variance > config.STAGGER_SWAY_THRESH or path_deviation > config.STAGGER_PATH_THRESH:
        return True

    return False