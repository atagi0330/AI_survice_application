import subprocess
import sys
import time
import os
import signal

def main():
    # ==========================================
    # ★ 同時起動するファイル名
    # ==========================================
    # 1. AI統合システム (カメラ映像: 転倒/ふらつき/顔認証)
    script_camera = "main_integrated.py"
    
    # 2. SwitchBot 温湿度監視 (コンソール出力)
    script_sensor = "switchbot_heat_stroke.py"

    # ファイルの存在確認
    if not os.path.exists(script_camera):
        print(f"エラー: {script_camera} が見つかりません。")
        print("直前に作成した統合コードを 'main_integrated.py' という名前で保存してください。")
        return
    
    if not os.path.exists(script_sensor):
        print(f"エラー: {script_sensor} が見つかりません。")
        print("SwitchBotのコードを 'switchbot_heat_stroke.py' という名前で保存してください。")
        return

    print("==================================================")
    print("   AI 工場安全管理システム & 環境モニタリング")
    print("           システムを一括起動します")
    print("==================================================")
    print("終了するには、この画面で [Ctrl+C] を押してください。")
    print("--------------------------------------------------")
    time.sleep(2)

    processes = []

    try:
        # --- 1. SwitchBot センサー監視 (バックグラウンド) ---
        print(f"起動中: {script_sensor} (環境センサ)...")
        # SwitchBotは通信待ちがあるため先に起動
        p_sensor = subprocess.Popen([sys.executable, script_sensor])
        processes.append(p_sensor)
        
        time.sleep(1) # ログが混ざらないように少し待機

        # --- 2. AI カメラシステム (メイン画面) ---
        print(f"起動中: {script_camera} (AIカメラ)...")
        p_camera = subprocess.Popen([sys.executable, script_camera])
        processes.append(p_camera)

        # 両方のプロセスが終了するのを待機
        # (片方が落ちても、もう片方は動き続けます)
        p_camera.wait()
        p_sensor.wait()

    except KeyboardInterrupt:
        print("\n\n!!! システム停止信号を受信しました !!!")
        print("すべてのプロセスを安全に終了しています...")
        
        # 起動したプロセスをすべて強制終了
        for p in processes:
            try:
                # Windows用強制終了コマンド
                if os.name == 'nt':
                    subprocess.call(['taskkill', '/F', '/T', '/PID', str(p.pid)])
                else:
                    p.terminate()
            except Exception:
                pass
        
        print("システムを完全に停止しました。")

if __name__ == "__main__":
    main()