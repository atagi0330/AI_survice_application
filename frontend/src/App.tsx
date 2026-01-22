import React, { useEffect, useMemo, useState } from 'react'
import { getAllLogs, getPersonLogs, getState } from './api'
import type { LogItem, Person, StateResponse } from './types'
import Modal from './components/Modal'

function tempLevel(temp: number): 'normal' | 'caution' | 'danger' {
  if (temp < 30) return 'normal'
  if (temp < 35) return 'caution'
  return 'danger'
}

function recommendations(temp: number, logs: LogItem[], persons: Person[]): string[] {
  const rec: string[] = []
  const lvl = tempLevel(temp)

  if (lvl === 'caution') {
    rec.push('温度が高めです。社員を休憩させるか、水分を取らせる必要があります。')
  } else if (lvl === 'danger') {
    rec.push('温度が高いので社員を休憩させるか、水分を取らせる必要があります。')
  }

  // Focus risk heuristic: any recent FOCUS log
  const hasFocus = logs.some(l => l.kind === 'FOCUS')
  if (hasFocus) {
    rec.push('集中力が低下している可能性がある社員がいます。気を付けて下さい。')
  }

  const hasFall = logs.some(l => l.kind === 'FALL')
  if (hasFall) {
    rec.push('転倒を検知しました。すぐに社員のもとへ向かい確認を行ってください。')
  }

  if (rec.length === 0) rec.push('現時点で緊急の推奨行動はありません。')
  return rec
}

function PhotoBox({ person, onClick }: { person: Person; onClick: () => void }) {
  const src = person.photo_url ?? null
  return (
    <button className="photoBtn" onClick={onClick} title="クリックすると個人ログを表示します">
      {src ? (
        <img className="photoImg" src={src} alt={`${person.name}の写真`} />
      ) : (
        <div className="photoPlaceholder">写真</div>
      )}
    </button>
  )
}

function PersonCard({ person, onPhotoClick }: { person: Person; onPhotoClick: () => void }) {
  return (
    <div className="personCard">
      <div className="personName">{person.name}</div>
      {/* 要件：名前の下の顔写真部分をクリックするとログ表示 */}
      <PhotoBox person={person} onClick={onPhotoClick} />
      <div className="personMeta">
        <div>ステータス：{person.status}</div>
        <div>緊急ステータス：{person.emergency_status}</div>
      </div>
    </div>
  )
}

function LogList({ logs, onPick }: { logs: LogItem[]; onPick: (l: LogItem) => void }) {
  return (
    <div className="logList">
      {logs.length === 0 && <div className="muted">ログはありません。</div>}
      {logs.map((l) => (
        <button key={l.id} className={`logRow ${l.kind.toLowerCase()}`} onClick={() => onPick(l)}>
          <div className="logMsg">{l.message}</div>
        </button>
      ))}
    </div>
  )
}

export default function App() {
  const [state, setState] = useState<StateResponse | null>(null)
  const [logs, setLogs] = useState<LogItem[]>([])

  const [personModalOpen, setPersonModalOpen] = useState(false)
  const [selectedPerson, setSelectedPerson] = useState<Person | null>(null)
  const [personLogs, setPersonLogs] = useState<LogItem[]>([])

  const [logModalOpen, setLogModalOpen] = useState(false)
  const [selectedLog, setSelectedLog] = useState<LogItem | null>(null)

  useEffect(() => {
    let alive = true
    ;(async () => {
      const st = await getState()
      const lg = await getAllLogs()
      if (!alive) return
      setState(st)
      // newest first
      setLogs([...lg].sort((a, b) => (a.timestamp < b.timestamp ? 1 : -1)))
    })()
    return () => {
      alive = false
    }
  }, [])

  const allPersons = useMemo(() => {
    if (!state) return []
    return [...state.persons_area_a, ...state.persons_area_b]
  }, [state])

  const recs = useMemo(() => {
    if (!state) return ['読み込み中...']
    return recommendations(state.temperature_c, logs, allPersons)
  }, [state, logs, allPersons])

  const level = state ? tempLevel(state.temperature_c) : 'normal'

  const openPersonLogs = async (p: Person) => {
    setSelectedPerson(p)
    const pl = await getPersonLogs(p.id)
    setPersonLogs([...pl].sort((a, b) => (a.timestamp < b.timestamp ? 1 : -1)))
    setPersonModalOpen(true)
  }

  const openLogDetail = (l: LogItem) => {
    setSelectedLog(l)
    setLogModalOpen(true)
  }

  return (
    <div className="page">
      <div className="title">管理者ダッシュボード</div>

      <div className="layout">
        {/* LEFT COLUMN */}
        <div className="leftCol">
          <div className={`card tempCard ${level}`}>
            <div className="cardLabel">温度</div>
            <div className="bigValue">{state ? state.temperature_c.toFixed(1) : '--'}°C</div>
            <div className="smallNote">
              {level === 'normal' && '平常（30度以下）'}
              {level === 'caution' && '注意（30〜35度）'}
              {level === 'danger' && '危険（35度以上）'}
            </div>
          </div>

          <div className="card">
            <div className="cardLabel">湿度</div>
            <div className="bigValue">{state ? `${state.humidity_pct.toFixed(0)}%` : '--'}</div>
          </div>

          <div className="card grow">
            <div className="cardLabel">推奨行動</div>
            <ul className="recList">
              {recs.map((r, i) => <li key={i}>{r}</li>)}
            </ul>
          </div>

          {/* 要件：左下ログ部分はクリックで写真等が表示 */}
          <div className="card logCard">
            <div className="cardLabel">ログ</div>
            <LogList logs={logs} onPick={openLogDetail} />
          </div>
        </div>

        {/* RIGHT COLUMN */}
        <div className="rightCol">
          <div className="area">
            <div className="areaTitle">エリアA</div>
            <div className="peopleRow">
              {state?.persons_area_a.map((p) => (
                <PersonCard key={p.id} person={p} onPhotoClick={() => openPersonLogs(p)} />
              ))}
            </div>
          </div>

          <div className="area">
            <div className="areaTitle">エリアB</div>
            <div className="peopleRow compact">
              {state?.persons_area_b.map((p) => (
                <div key={p.id} className="miniSlot">
                  <button className="miniPhoto" onClick={() => openPersonLogs(p)} title="クリックすると個人ログを表示します">
                    <span className="miniText">写真</span>
                  </button>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Person logs modal */}
      <Modal
        title={selectedPerson ? `${selectedPerson.name} のログ` : '個人ログ'}
        isOpen={personModalOpen}
        onClose={() => setPersonModalOpen(false)}
      >
        <div className="modalSection">
          <div className="muted">※「写真」部分をクリックすると、この個人のログ一覧が表示されます。</div>
        </div>
        <LogList logs={personLogs} onPick={openLogDetail} />
      </Modal>

      {/* Log detail modal */}
      <Modal
        title={selectedLog ? `ログ詳細（${selectedLog.kind}）` : 'ログ詳細'}
        isOpen={logModalOpen}
        onClose={() => setLogModalOpen(false)}
      >
        {!selectedLog ? null : (
          <div className="logDetail">
            <div className="logDetailMsg">{selectedLog.message}</div>
            <div className="snapWrap">
              {selectedLog.snapshot_url ? (
                <img
                  className="snapImg"
                  src={selectedLog.snapshot_url.startsWith('http') ? selectedLog.snapshot_url : `http://localhost:8000${selectedLog.snapshot_url}`}
                  alt="ログ発生時の写真"
                />
              ) : (
                <div className="snapPlaceholder">写真は未登録です</div>
              )}
            </div>
          </div>
        )}
      </Modal>
    </div>
  )
}
