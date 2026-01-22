# 管理者ダッシュボード UI（要件対応版）

## UI要件（対応済み）
- 名前の下の「顔写真」部分をクリック → その個人のステータスログ一覧を表示
- 左下のログ（一覧）をクリック → そのログ発生時の写真などを表示
- 温度カードは背景色で危険度表示
  - 30度以下：平常（緑）
  - 30-35度：注意（黄）
  - 35度以上：危険（赤）
- 推奨行動（例文ベース）を温度・ログ内容から自動生成（異常時は該当者名を表示）

## 起動方法

### Backend（FastAPI）
```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

確認：
- http://localhost:8000/health
- http://localhost:8000/api/state
- http://localhost:8000/api/logs

### Frontend（Vite + React）
```powershell
cd frontend
npm install
npm run dev
```

- http://localhost:5173

## 補足
- 写真はログの `snapshot_url` があれば表示されます（例：/static/snapshots/...）。
- 個人の顔写真を表示したい場合は、Person の `photo_url` を埋めれば表示されます。
