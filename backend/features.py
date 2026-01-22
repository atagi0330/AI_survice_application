# features.py
import time
import math
import numpy as np
from collections import deque
import config

class PersonState:
    def __init__(self):
        self.state = "Normal"
        self.start_time = time.time()
        self.status_color = (0, 255, 0) # Green
        self.action_message = "Normal"
        self.alert_active = False

        # ふらつき検知用の履歴
        self.body_angle_hist = deque(maxlen=config.STAGGER_HISTORY_LEN)
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
        self.action_message = "Normal"
        self.alert_active = False


def get_point(kpts, idx):
    if kpts[idx][2] < config.KEYPOINT_CONF_TH:
        return None
    return kpts[idx][:2]

def calculate_vertical_angle(p1, p2):
    """
    2点間のベクトルが垂直線となす角度(度)を計算
    0度=垂直, 90度=水平
    """
    dx = abs(p1[0] - p2[0])
    dy = abs(p1[1] - p2[1])
    if dy == 0: return 90.0
    angle_rad = math.atan2(dx, dy)
    return math.degrees(angle_rad)

def is_sitting_pose(kpts):
    """
    座っているか判定する
    判定基準: 太もも（腰-膝）が水平に近い場合
    """
    l_hip = get_point(kpts, 11)
    r_hip = get_point(kpts, 12)
    l_knee = get_point(kpts, 13)
    r_knee = get_point(kpts, 14)

    # 少なくとも片足が見えていること
    sitting_legs = 0
    visible_legs = 0

    # 左足判定
    if l_hip is not None and l_knee is not None:
        visible_legs += 1
        angle = calculate_vertical_angle(l_hip, l_knee)
        # 太ももが垂直から大きく傾いている(45度以上)なら座っているとみなす
        if angle > config.SITTING_THIGH_ANGLE_THRESH:
            sitting_legs += 1
    
    # 右足判定
    if r_hip is not None and r_knee is not None:
        visible_legs += 1
        angle = calculate_vertical_angle(r_hip, r_knee)
        if angle > config.SITTING_THIGH_ANGLE_THRESH:
            sitting_legs += 1

    # 両足とも見えない(机の下など)場合も、誤検知防止のため「座っている」扱いにすることが多いが、
    # ここでは「足が見えていて、かつ横になっている」場合のみTrueとする
    # ※足が見えない＝ふらつき検知もしない、という安全策を取るならここでTrueを返しても良い
    if visible_legs > 0 and sitting_legs == visible_legs:
        return True
    
    return False

def check_fall_pose(kpts, bbox=None):
    """
    転倒判定ロジック (座り誤検知対策済み)
    """
    l_sh = get_point(kpts, 5)
    r_sh = get_point(kpts, 6)
    l_hip = get_point(kpts, 11)
    r_hip = get_point(kpts, 12)

    if l_sh is None or r_sh is None or l_hip is None or r_hip is None:
        return False

    # --- ★ 対策: 上半身(Torso)の角度チェック ---
    mid_sh = (l_sh + r_sh) / 2
    mid_hip = (l_hip + r_hip) / 2
    
    torso_angle = calculate_vertical_angle(mid_sh, mid_hip)
    
    # 上半身が起きている(角度が閾値未満)なら、絶対に転倒ではない
    # 座っている人はここで弾かれる (角度0〜20度くらいのため)
    if torso_angle < config.FALL_TORSO_ANGLE_THRESH:
        return False

    # 以下、従来のBBox判定（念のため残すが、上の角度判定でほぼ解決する）
    if bbox is not None:
        _, y1, _, y2 = bbox
        h = float(max(1, y2 - y1))
        w = float(max(1, bbox[2] - bbox[0]))
    else:
        # fallback
        return True 

    if h < config.FALL_MIN_HEIGHT: return False

    # アスペクト比チェック
    aspect = w / h
    # 上半身も倒れていて、かつ横長なら転倒
    if aspect > config.FALL_ASPECT_RATIO:
        return True

    return False

def check_bad_posture(kpts):
    # 座っているときも姿勢判定はしたいので、そのままにする
    # ただし、座っていると thigh_dy が小さくなり判定が不安定になる可能性があるため
    # 立位前提のロジックなら is_sitting_pose チェックを入れても良い
    l_sh = get_point(kpts, 5)
    l_hip = get_point(kpts, 11)
    l_knee = get_point(kpts, 13)

    if l_sh is not None and l_hip is not None and l_knee is not None:
        thigh_dy = abs(l_hip[1] - l_knee[1])
        torso_dy = abs(l_sh[1] - l_hip[1])
        
        # 座っている(thigh_dyが小さい)場合は除外、またはロジックを変える
        if thigh_dy < 10: return False # ゼロ除算/座り対策

        if (torso_dy / thigh_dy < config.POSTURE_RATIO_LIMIT):
            return True
    return False

def check_staggering(kpts, state):
    """
    ふらつき判定 (座り誤検知対策済み)
    """
    # ★ 対策: 座っている人はふらつき検知しない
    if is_sitting_pose(kpts):
        return False

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
    
    # A. 体幹の傾き (Body Sway)
    dx = mid_shoulder[0] - mid_hip[0]
    dy = mid_shoulder[1] - mid_hip[1]
    
    if dy != 0:
        angle = np.degrees(np.arctan2(dx, -dy)) 
    else:
        angle = 0
    state.body_angle_hist.append(angle)

    # B. 重心のブレ (Zigzag Path)
    state.center_x_hist.append(mid_hip[0])

    # 判定
    if len(state.body_angle_hist) < config.STAGGER_HISTORY_LEN:
        return False

    angle_variance = np.std(state.body_angle_hist)
    
    x_history = list(state.center_x_hist)
    smooth_x = np.convolve(x_history, np.ones(5)/5, mode='valid') 
    current_x_segment = x_history[-len(smooth_x):]
    path_deviation = np.mean(np.abs(np.array(current_x_segment) - smooth_x))

    if angle_variance > config.STAGGER_SWAY_THRESH or path_deviation > config.STAGGER_PATH_THRESH:
        return True

    return False