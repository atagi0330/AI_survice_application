# 管理者ダッシュボード UI（要件対応版）

作業者の異常検知（転倒・ふらつき・姿勢不良・居眠り等）と、温湿度環境監視を統合表示する管理者向けダッシュボードです。  
（Backend: FastAPI / Frontend: React + Vite）

---

## UI要件（対応済み）

- 名前の下の「顔写真」部分をクリック → **その個人のステータスログ一覧を表示**
- 左下のログ（一覧）をクリック → **そのログ発生時の写真などを表示**
- 温度カードは背景色で危険度表示
  - **30度以下：平常（緑）**
  - **30-35度：注意（黄）**
  - **35度以上：危険（赤）**
- 推奨行動（例文ベース）を温度・ログ内容から自動生成
  - 異常時は**該当者名を表示**

---

## 起動方法

### Backend（FastAPI）

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
確認：

http://localhost:8000/health

http://localhost:8000/api/state

http://localhost:8000/api/logs
```

---

### Frontend（Vite + React）
cd frontend
npm install
npm run dev
http://localhost:5173

---

###API（フロント利用）
GET /health

GET /api/state

GET /api/logs

GET /api/person/{personId}/logs

WS /ws/events（リアルタイム更新）

### データ表示仕様
温度表示
温度カードの色は現在温度から自動判定

temp <= 30 → 平常（緑）

30 < temp < 35 → 注意（黄）

temp >= 35 → 危険（赤）

### 推奨行動
温度帯 + 最新ログ種別（FALL / STAGGER / DROWSY / SLEEP など）に応じて文言を自動生成

人物イベントがある場合は対象者名を文中に表示

ログ詳細
ログに snapshot_url がある場合、詳細モーダルで画像を表示

snapshot_url が無い場合はテキスト情報のみ表示

---

###補足
写真はログの snapshot_url があれば表示されます（例：/static/snapshots/...）。

個人の顔写真を表示したい場合は、Person の photo_url を設定してください。

VITE_API_BASE を設定するとフロントの接続先 API を変更できます（未設定時は http://localhost:8000）。

例（frontend/.env）:

VITE_API_BASE=http://localhost:8000
ディレクトリ構成（例）
backend/
  app/
    main.py
  requirements.txt

frontend/
  src/
    App.tsx
    api.ts
    components/
      Modal.tsx
  package.json
  vite.config.ts
トラブルシューティング
APIに接続できない
Backend が :8000 で起動しているか確認

VITE_API_BASE のURLを確認

CORS設定と WebSocket エンドポイントを確認

画像が表示されない
snapshot_url が有効なパスか確認

静的ファイル配信（/static/...）設定を確認

フロントに最新状態が反映されない
GET /api/state のレスポンス確認

WebSocket (/ws/events) 接続状態を確
