from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from datetime import datetime, timedelta
from pydantic import BaseModel
from typing import List, Dict, Optional

app = FastAPI(title="Health Dashboard Mock API")

# Allow local dev frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

class Person(BaseModel):
    id: int
    name: str
    photo_url: Optional[str] = None
    status: str
    emergency_status: str

class LogItem(BaseModel):
    id: str
    timestamp: str  # ISO
    person_id: int
    person_name: str
    message: str
    snapshot_url: Optional[str] = None
    kind: str  # FALL, FOCUS, INFO

class StateResponse(BaseModel):
    temperature_c: float
    humidity_pct: float
    persons_area_a: List[Person]
    persons_area_b: List[Person]

def _now():
    return datetime.now()

# Mock data
PERSONS_A = [
    Person(id=1, name="田中 太郎", status="正常", emergency_status="無"),
    Person(id=2, name="佐藤 花子", status="検知なし", emergency_status="転倒の可能性"),
]

PERSONS_B = [
    Person(id=3, name="鈴木 次郎", status="正常", emergency_status="無"),
    Person(id=4, name="高橋 美咲", status="検知なし", emergency_status="無"),
    Person(id=5, name="伊藤 健", status="正常", emergency_status="無"),
]

def build_logs():
    # Create a few logs with "1月16日..." style in message but keep ISO in timestamp too
    base = _now().replace(month=1, day=16, hour=17, minute=5, second=0, microsecond=0)
    l1 = LogItem(
        id="evt-001",
        timestamp=base.isoformat(),
        person_id=1,
        person_name="田中 太郎",
        kind="FALL",
        message="1月16日17:05 田中 太郎　田中 太郎の転倒を検知しました。",
        snapshot_url="/static/snapshots/fall_evt_001.png",
    )
    l2 = LogItem(
        id="evt-002",
        timestamp=(base - timedelta(minutes=25)).isoformat(),
        person_id=2,
        person_name="佐藤 花子",
        kind="FOCUS",
        message="1月16日16:40 佐藤 花子　集中力が低下している可能性があります。",
        snapshot_url="/static/snapshots/focus_evt_002.png",
    )
    l3 = LogItem(
        id="evt-003",
        timestamp=(base - timedelta(minutes=40)).isoformat(),
        person_id=3,
        person_name="鈴木 次郎",
        kind="INFO",
        message="1月16日16:25 鈴木 次郎　特記事項はありません。",
        snapshot_url=None,
    )
    return [l1, l2, l3]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/api/state", response_model=StateResponse)
def get_state():
    return StateResponse(
        temperature_c=28.3,
        humidity_pct=56.0,
        persons_area_a=PERSONS_A,
        persons_area_b=PERSONS_B,
    )

@app.get("/api/logs", response_model=List[LogItem])
def get_logs():
    return build_logs()

@app.get("/api/person/{person_id}/logs", response_model=List[LogItem])
def get_person_logs(person_id: int):
    return [l for l in build_logs() if l.person_id == person_id]
