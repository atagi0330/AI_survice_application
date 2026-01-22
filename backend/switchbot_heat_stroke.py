import time
import hashlib
import hmac
import base64
import uuid
import requests
import math
from datetime import datetime
# ==========================================
# ★設定：ここを書き換えてください
# ==========================================
TOKEN = '88b2dbf0cb360bfa946e5ce1739d844f9ade5beecf877098aed2cbb8eb41e0c4fbd08f5c46a71b3393dc3b533a33f8a2'
SECRET = '513d0bd2b33758d0ea6452f3203500ac'

# ------------------------------------------
# API認証ヘッダー作成関数
# ------------------------------------------
def make_header():
    nonce = uuid.uuid4()
    t = int(round(time.time() * 1000))
    string_to_sign = '{}{}{}'.format(TOKEN, t, nonce)
    
    string_to_sign_b = bytes(string_to_sign, 'utf-8')
    secret_b = bytes(SECRET, 'utf-8')
    sign = base64.b64encode(hmac.new(secret_b, msg=string_to_sign_b, digestmod=hashlib.sha256).digest())
    
    return {
        'Authorization': TOKEN,
        't': str(t),
        'sign': str(sign, 'utf-8'),
        'nonce': str(nonce)
    }

# ------------------------------------------
# 計算ロジック (WBGT, VPDなど)
# ------------------------------------------
def calc_metrics(temp, hum):
    # 飽和水蒸気圧 [hPa]
    es = 6.1078 * math.pow(10, (7.5 * temp) / (temp + 237.3))
    # 実在水蒸気圧 [hPa]
    e = es * (hum / 100.0)
    # 容積絶対湿度 [g/m^3]
    abs_hum = (217 * e) / (temp + 273.15)
    # VPD [kPa]
    vpd = (es - e) / 10.0
    # WBGT (簡易推定) [℃]
    wbgt = 0.567 * temp + 0.393 * e + 3.94
        
    alert = "安全"
    if wbgt >= 35: alert = "【危険】熱中症の危険性が高いです！今すぐに休憩と水分補給を！"
    elif wbgt >= 30: alert = "【注意】体調に気を付けて適度な休憩を取ってください！"
    elif wbgt <= 29: alert = "【通常】"
    return {"abs_hum": abs_hum, "vpd": vpd, "wbgt": wbgt, "alert": alert}

# ------------------------------------------
# データ取得・表示処理 (1回分)
# ------------------------------------------
def fetch_data():
    print("SwitchBot APIに接続中... 温湿度計を探しています...")
    
    # 1. デバイスリストを取得
    url_devices = "https://api.switch-bot.com/v1.1/devices"
    headers = make_header()
    
    try:
        res = requests.get(url_devices, headers=headers)
        res.raise_for_status()
        device_list = res.json()['body']['deviceList']
        
        targets = [d for d in device_list if d['deviceType'] in ["Meter", "MeterPlus", "WoIOSensor", "Hub 2"]]
        
        if not targets:
            print("エラー: 温湿度計が見つかりませんでした。")
            return

        print(f"{len(targets)} 台の温湿度計が見つかりました。")

        # 2. 見つかった全デバイスのデータを取得して表示
        for device in targets:
            d_id = device['deviceId']
            d_name = device['deviceName']
            
            print(f"--- 取得中: {d_name} ---")
            
            # ステータス取得
            url_status = f"https://api.switch-bot.com/v1.1/devices/{d_id}/status"
            headers_status = make_header() # ヘッダー再生成(time更新)
            
            res_status = requests.get(url_status, headers=headers_status)
            res_status.raise_for_status()
            status_body = res_status.json()['body']
            
            temp = status_body.get('temperature', 0)
            hum = status_body.get('humidity', 0)
            
            if temp == 0 and hum == 0:
                print("  [!] データが 0 です。クラウドサービス設定を確認してください。")
                continue

            # 計算実行
            metrics = calc_metrics(temp, hum)
            
            # 表示
            print(f"  温度     : {temp} ℃")
            print(f"  湿度     : {hum} %")
            print(f"  WBGT     : {metrics['wbgt']:.1f} ℃ ({metrics['alert']})")
            print(f"  VPD      : {metrics['vpd']:.2f} kPa")
            print(f"  絶対湿度 : {metrics['abs_hum']:.1f} g/m^3")
            print("")
            
    except Exception as e:
        print(f"エラーが発生しました: {e}")

# ------------------------------------------
# メインループ
# ------------------------------------------
def main():
    print("=== 温湿度監視プログラムを開始します (Ctrl+C で終了) ===")
    try:
        while True:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{now}] データ更新開始")
            
            fetch_data()
            
            print("次の更新まで待機中 (60秒)...")
            time.sleep(60)
            
    except KeyboardInterrupt:
        print("\nプログラムを終了します。")

if __name__ == "__main__":
    main()