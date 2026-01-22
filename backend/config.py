# config.py
import torch

# デバイス設定
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 使用するYOLO Poseモデル
POSE_MODEL_PATH = 'yolo26m-pose.pt'

# --- 時間ベースのトリガー設定 (秒単位) ---
# 転倒: ほぼ即時に危険判定
DURATION_LIMIT_FALL = 0.3       

# 悪い姿勢: 長時間続いたら「ストレッチ推奨」
DURATION_LIMIT_POSTURE = 5.0    

# ふらつき: 一定時間続いたら「休憩・水分補給推奨」
DURATION_LIMIT_STAGGER = 1.0    

# --- ★ 【追加】ふらつき検知 (高度版) パラメータ ---
# 過去何フレームを解析するか (30fpsなら30=約1秒)
STAGGER_HISTORY_LEN = 30   

# 体幹の揺れ(角度の分散)のしきい値
# 数値を小さくすると敏感になり、大きくすると鈍感になります
STAGGER_SWAY_THRESH = 4.5 

# 歩行軌道のズレ(ピクセル)のしきい値
# 数値を小さくすると敏感になり、大きくすると鈍感になります
STAGGER_PATH_THRESH = 7.0 

# --- 姿勢・検知パラメータ ---
# 垂直距離の比率 (肩-腰 / 腰-膝)。これが小さいと前屈み
POSTURE_RATIO_LIMIT = 0.6      

# --- 転倒判定パラメータ ---
# バウンディングボックスの高さが小さすぎる場合はノイズとみなす
FALL_MIN_HEIGHT = 80           # [px]
# 横向き（寝転び）とみなすアスペクト比 (幅 / 高さ)
FALL_ASPECT_RATIO = 1.3        
# 肩と腰の縦方向の距離 / 身長（bbox 高さ）
FALL_VERTICAL_RATIO = 0.35     

# キーポイント検出の最低信頼度
KEYPOINT_CONF_TH = 0.3         

# 画像保存設定
LOG_DIR = "logs"
NORMAL_INTERVAL = 60  # 通常時の保存間隔(秒)
ALERT_INTERVAL = 2    # 異常時の保存間隔(秒)
NORMAL_MAX_FILES = 100