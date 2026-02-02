import React, { useEffect, useMemo, useState } from 'react'
import type { LogItem, Person, StateResponse } from './types'
import { apiBase, getLogs, getPersonLogs, getState, resolveSnapshotUrl } from './api'
import Modal from './components/Modal'

function uniq(xs: string[]) { return Array.from(new Set(xs)).filter(Boolean) }

function tempLevel(t: number) {
  if (t < 30) return 'normal'
  if (t < 35) return 'caution'
  return 'danger'
}

function tempBg(t: number) {
  const lvl = tempLevel(t)
  if (lvl === 'normal') return 'tempNormal'
  if (lvl === 'caution') return 'tempCaution'
  return 'tempDanger'
}

function buildRecommendations(temp: number, persons: Person[]): string[] {
  const rec: string[] = []

  // const lvl = tempLevel(temp)
  // if (lvl === 'caution') rec.push('温度が高いので、従業員に水分や休息をとらせる必要があります。')
  // if (lvl === 'danger') rec.push('熱中症の危険があります。今すぐ休息をとらせる必要があります。')

  function getAction(p: Person): string | null {
    // 1. Emergency Status (Body)
    // 1. Emergency Status (Body)
    const em = (p.emergency_status ?? '').toUpperCase()
    if (em.includes('FALL') || em.includes('転倒')) return `${p.name}さんが転倒を検知しました。今すぐに${p.name}さんのもとへ向かってください。`
    if (em.includes('STAGGER') || em.includes('ふらつき')) return `${p.name}さんがふらついています。水分補給と休憩を取らせてください。`

    // 2. Status (Face)
    const st = (p.status ?? '').toUpperCase()
    if (st.includes('SLEEP') || st.includes('居眠り')) return `${p.name}さんが居眠りしている可能性があります。危険なため、作業を中止させ今すぐに起こしに行ってください。`
    if (st.includes('DROWSY') || st.includes('眠気')) return `${p.name}さんに眠気がある可能性があります。注意してください。`
    if (st.includes('YAWN') || st.includes('あくび')) return `${p.name}さんの集中が切れているため、ストレッチまたは水分補給などを行ってください。`

    // 3. その他異常
    if ((em && em !== '無' && em !== 'NORMAL') || (st && st !== '正常' && st !== 'NORMAL')) {
      return `${p.name} : 異常検出 (${em || st}) - 確認してください`
    }
    return null
  }

  for (const p of persons) {
    const act = getAction(p)
    if (act) {
      rec.push(act)
    }
  }

  if (!rec.length) rec.push('現時点で緊急の推奨行動はありません。')
  return rec
}

function LogRow({ l, onClick }: { l: LogItem; onClick: () => void }) {
  const kind = (l.kind ?? 'INFO').toUpperCase()
  const cls =
    kind === 'FALL' || kind === 'SLEEP' ? 'logRow fall' :
      kind === 'FOCUS' || kind === 'YAWN' || kind === 'DROWSY' ? 'logRow focus' :
        kind === 'STAGGER' || kind === 'POSTURE' ? 'logRow warn' :
          'logRow info'

  return (
    <button className={cls} onClick={onClick}>
      <div className="logMsg">{l.message}</div>
    </button>
  )
}

function PersonCard({ p, onPhotoClick }: { p: Person; onPhotoClick: () => void }) {
  const emer = (p.emergency_status ?? '')
  const warn = emer.includes('転倒') || emer.includes('危険') || emer.includes('居眠り')
  const src = resolveSnapshotUrl(p.photo_url ?? null)

  return (
    <div className={'personCard' + (warn ? ' personWarn' : '')}>
      <div className="personName">{p.name}</div>
      <button className="photo" onClick={onPhotoClick} title="クリックすると個人ログを表示します">
        {src ? <img className="photoImg" src={src} alt={p.name} /> : <div className="photoInner"></div>}
      </button>
      <div className="meta">
        <div>ステータス：{p.status}</div>
        <div>緊急ステータス：{(p.emergency_status === '無' || !p.emergency_status) ? '異常なし' : p.emergency_status}</div>
      </div>
    </div>
  )
}

export default function App() {
  const [state, setState] = useState<StateResponse | null>(null)
  const [logs, setLogs] = useState<LogItem[]>([])
  const [loading, setLoading] = useState(true)

  const [personModalOpen, setPersonModalOpen] = useState(false)
  const [selectedPerson, setSelectedPerson] = useState<Person | null>(null)
  const [personLogs, setPersonLogs] = useState<LogItem[]>([])

  const [logModalOpen, setLogModalOpen] = useState(false)
  const [selectedLog, setSelectedLog] = useState<LogItem | null>(null)

  const temp = state?.temperature_c ?? 0
  const hum = state?.humidity_pct ?? 0
  const allPersons = useMemo(() => [...(state?.persons_area_a ?? []), ...(state?.persons_area_b ?? [])], [state])
  const recs = useMemo(() => buildRecommendations(temp, allPersons), [temp, allPersons])

  const refresh = async () => {
    try {
      const [s, l] = await Promise.all([getState(), getLogs()])
      setState(s)
      setLogs(l)
      setLoading(false)
    } catch {
      // keep previous; show loading only first time
      setLoading(false)
    }
  }

  useEffect(() => {
    refresh()

    // WebSocket (real-time) — fallback to polling if not available.
    const wsUrl = apiBase().replace('http://', 'ws://').replace('https://', 'wss://') + '/ws/events'
    let ws: WebSocket | null = null
    let pingTimer: number | null = null
    let pollTimer: number | null = null

    const startPolling = () => {
      if (pollTimer) return
      pollTimer = window.setInterval(refresh, 1500)
    }

    try {
      ws = new WebSocket(wsUrl)
      ws.onopen = () => {
        // keepalive ping
        pingTimer = window.setInterval(() => {
          try { ws?.send('ping') } catch { }
        }, 15000)
      }
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data)
          if (msg.type === 'snapshot') {
            const st = msg.data?.state as StateResponse
            const evs = msg.data?.events as LogItem[]
            if (st) setState(st)
            if (evs) setLogs(evs)
          } else if (msg.type === 'person_state') {
            // Merge into state
            setState((prev) => {
              if (!prev) return prev
              const d = msg.data
              const p: Person = { id: d.person_id, name: d.person_name, status: d.status, emergency_status: d.emergency_status, photo_url: d.photo_url }
              const area = (d.area ?? ((p.id % 2 === 1) ? 'A' : 'B')) as 'A' | 'B'
              const up = (arr: Person[]) => {
                const idx = arr.findIndex(x => x.id === p.id)
                if (idx >= 0) {
                  const copy = arr.slice()
                  copy[idx] = { ...copy[idx], ...p }
                  return copy
                }
                return [...arr, p]
              }
              return {
                ...prev,
                persons_area_a: area === 'A' ? up(prev.persons_area_a) : prev.persons_area_a.filter(x => x.id !== p.id),
                persons_area_b: area === 'B' ? up(prev.persons_area_b) : prev.persons_area_b.filter(x => x.id !== p.id),
              }
            })
          } else if (msg.type === 'event') {
            const e = msg.data as LogItem
            setLogs((prev) => [e, ...prev].slice(0, 300))
          } else if (msg.type === 'environment') {
            setState((prev) => prev ? { ...prev, temperature_c: msg.data.temperature_c, humidity_pct: msg.data.humidity_pct } : prev)
          }
        } catch {
          // ignore
        }
      }
      ws.onerror = () => { startPolling() }
      ws.onclose = () => { startPolling() }
    } catch {
      startPolling()
    }

    // also poll occasionally to keep state consistent even with WS
    startPolling()

    return () => {
      if (pingTimer) window.clearInterval(pingTimer)
      if (pollTimer) window.clearInterval(pollTimer)
      try { ws?.close() } catch { }
    }
  }, [])

  const openLogDetail = (l: LogItem) => {
    setSelectedLog(l)
    setLogModalOpen(true)
  }

  const openPersonLogs = async (p: Person) => {
    setSelectedPerson(p)
    try {
      const pl = await getPersonLogs(p.id)
      setPersonLogs(pl)
    } catch {
      setPersonLogs(logs.filter(x => x.person_id === p.id))
    }
    setPersonModalOpen(true)
  }

  const personsA = state?.persons_area_a ?? []
  const personsB = state?.persons_area_b ?? []

  return (
    <div className="page">
      <div className="title">管理者ダッシュボード</div>

      <div className="layout">
        {/* LEFT */}
        <div className="leftCol">
          <div className={'box ' + tempBg(temp)}>
            <div className="boxText">
              {/* 温度部分：ここを div で囲んで改行禁止(nowrap)にします */}
              <div style={{ whiteSpace: 'nowrap' }}>
                温度：{loading ? '--' : temp.toFixed(1)}℃
              </div>

              {/* メッセージ部分：span を div に変更します */}
              <div style={{ marginLeft: '30px', fontSize: '30px', fontWeight: 'normal' }}>
                {temp < 30 ? '異常なし' :
                  temp < 35 ? '注意。適度に水分補給を補給してください。' :
                    '危険。全従業員水分補給及び休憩を取ってください‼'}
              </div>
            </div>
          </div>

          <div className="box">
            <div className="boxText">
              <div style={{ whiteSpace: 'nowrap' }}>
                湿度：{loading ? '--' : hum.toFixed(0)}%
              </div>
              <div style={{ marginLeft: '30px', fontSize: '30px', fontWeight: 'normal' }}>
                {hum < 40 ? '加湿器を稼働し、こまめな水分補給をしてください。' :
                  hum > 60 ? '除湿・換気を行い、休憩を増やしてください。熱中症に注意してください‼' :
                    '異常なし'}
              </div>
            </div>
          </div>

          <div className="bigBox">
            <div className="bigTitle">推奨行動</div>
            <ul className="rec">
              {recs.map((r, i) => <li key={i}>{r}</li>)}
            </ul>
          </div>


        </div>

        {/* RIGHT */}
        <div className="rightCol">
          <div className="area">
            <div className="areaTitle">現在の状況</div>
            <div className="areaInner">
              {allPersons.length === 0 ? <div className="muted">データ待ち...</div> : null}
              {allPersons.map(p => (
                <PersonCard key={p.id} p={p} onPhotoClick={() => openPersonLogs(p)} />
              ))}
            </div>
          </div>

          <div className="area">
            <div className="areaTitle">ログ</div>
            <div className="logList">
              {logs.length === 0 ? <div className="muted">ログはまだありません。</div> : null}
              {logs.map(l => <LogRow key={l.id} l={l} onClick={() => openLogDetail(l)} />)}
            </div>
          </div>
        </div>
      </div>

      <Modal
        title={selectedPerson ? `${selectedPerson.name} のログ` : '個人ログ'}
        open={personModalOpen}
        onClose={() => setPersonModalOpen(false)}
      >
        <div className="muted">※ログをクリックすると写真が表示されます。</div>
        <div className="logList">
          {personLogs.length === 0 ? <div className="muted">ログはまだありません。</div> : null}
          {personLogs.map(l => <LogRow key={l.id} l={l} onClick={() => openLogDetail(l)} />)}
        </div>
      </Modal>

      <Modal
        title={selectedLog ? 'ログ詳細' : 'ログ詳細'}
        open={logModalOpen}
        onClose={() => setLogModalOpen(false)}
      >
        {!selectedLog ? null : (
          <div>
            <div className="detailMsg">{selectedLog.message}</div>
            <div className="snap">
              {resolveSnapshotUrl(selectedLog.snapshot_url) ? (
                <img className="snapImg" src={resolveSnapshotUrl(selectedLog.snapshot_url)!} alt="ログ発生時の写真" />
              ) : (
                <div className="muted">写真は未登録です。</div>
              )}
            </div>
          </div>
        )}
      </Modal>

      <div className="footerMuted">API: {apiBase()}</div>
    </div>
  )
}
