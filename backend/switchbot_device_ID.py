import time
import hashlib
import hmac
import base64
import uuid
import requests
import json

# ==========================================
# ★ここにアプリから取ったキーを貼る
# ==========================================
token = '88b2dbf0cb360bfa946e5ce1739d844f9ade5beecf877098aed2cbb8eb41e0c4fbd08f5c46a71b3393dc3b533a33f8a2'
secret = '513d0bd2b33758d0ea6452f3203500ac'
# ==========================================

def make_header(token, secret):
    nonce = uuid.uuid4()
    t = int(round(time.time() * 1000))
    string_to_sign = '{}{}{}'.format(token, t, nonce)
    string_to_sign = bytes(string_to_sign, 'utf-8')
    secret = bytes(secret, 'utf-8')
    sign = base64.b64encode(hmac.new(secret, msg=string_to_sign, digestmod=hashlib.sha256).digest())
    return {
        'Authorization': token,
        't': str(t),
        'sign': str(sign, 'utf-8'),
        'nonce': str(nonce)
    }

def get_device_list():
    print("デバイスリストを取得中...")
    url = "https://api.switch-bot.com/v1.1/devices"
    headers = make_header(token, secret)
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        device_list = data['body']['deviceList']
        
        print("\n--- 発見されたデバイス ---")
        for device in device_list:
            d_type = device.get('deviceType')
            # 温湿度計関連のみ表示（Meter, WoIOSensorなど）
            if "Meter" in d_type or "Hub" in d_type or "Sensor" in d_type:
                print(f"名前: {device['deviceName']}")
                print(f"タイプ: {d_type}")
                print(f"ID: {device['deviceId']}") # ★このIDを使います
                print("-" * 20)
                
    except Exception as e:
        print(f"エラー: {e}")

if __name__ == "__main__":
    get_device_list()