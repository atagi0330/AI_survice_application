import subprocess
import sys
import time
import os

def main():
    # ==========================================
    # ★ 同時起動するファイル構成
    # ==========================================
    # 1. API中継サーバー (FrontendとBackendの架け橋)
    script_server = "server.py"
    
    # 2. SwitchBot 温湿度監視 (環境データ取得)
    script_sensor = "switchbot_heat_stroke.py"

    # 3. AI統合システム (カメラ映像: 転倒/ふらつき/顔認証/居眠り)
    script_camera = "main_integrated.py"

    # 4. フロントエンド
    frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")

    # ファイルの存在確認
    scripts = [script_server, script_sensor, script_camera]
    for script in scripts:
        if not os.path.exists(script):
            print(f"エラー: {script} が見つかりません。")
            return
    
    if not os.path.exists(frontend_dir):
        print(f"警告: フロントエンドディレクトリが見つかりません: {frontend_dir}")
        frontend_dir = None

    print("==================================================")
    print("   AI 工場安全管理システム & 環境モニタリング")
    print("        フロントエンド + バックエンド一括起動")
    print("==================================================")
    print("終了するには、この画面で [Ctrl+C] を押してください。")
    print("--------------------------------------------------")
    
    processes = []

    try:
        # --- 0. フロントエンドの起動 ---
        if frontend_dir:
            print("0/4 起動中: フロントエンド (Vite)...")
            p_frontend = subprocess.Popen(
                ["npm", "run", "dev"],
                cwd=frontend_dir,
                shell=True
            )
            processes.append(p_frontend)
            time.sleep(3)  # Viteの起動を待つ

        # --- 1. APIサーバーの起動 ---
        print(f"1/4 起動中: {script_server} (APIサーバー)...")
        p_server = subprocess.Popen([sys.executable, script_server])
        processes.append(p_server)
        time.sleep(2) # サーバーの立ち上がりを待つ

        # --- 2. SwitchBot センサー監視の起動 ---
        print(f"2/4 起動中: {script_sensor} (環境センサ)...")
        p_sensor = subprocess.Popen([sys.executable, script_sensor])
        processes.append(p_sensor)
        time.sleep(1)

        # --- 3. AI カメラシステムの起動 ---
        print(f"3/4 起動中: {script_camera} (AIカメラ映像解析)...")
        p_camera = subprocess.Popen([sys.executable, script_camera])
        processes.append(p_camera)

        print("\n>>> すべてのシステムが正常に起動しました。")
        print(">>> ブラウザで http://localhost:5173/ を開いてください。")

        # プロセスが終了するのを待機
        # メイン画面であるカメラシステムが閉じられたら終了とみなす
        p_camera.wait()

    except KeyboardInterrupt:
        print("\n\n!!! システム停止信号(Ctrl+C)を受信しました !!!")
    except Exception as e:
        print(f"\n予期せぬエラーが発生しました: {e}")
    finally:
        print("すべてのプロセスを安全に終了しています...")
        # 起動したすべてのプロセスを確実に停止させる
        for p in processes:
            try:
                if os.name == 'nt': # Windowsの場合
                    subprocess.call(['taskkill', '/F', '/T', '/PID', str(p.pid)])
                else: # Mac/Linuxの場合
                    p.terminate()
            except Exception:
                pass
        
        print("システムを完全に停止しました。お疲れ様でした。")

if __name__ == "__main__":
    main()