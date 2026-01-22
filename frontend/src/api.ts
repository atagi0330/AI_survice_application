import type { StateResponse, LogItem } from './types'

const DEFAULT_API = 'http://localhost:8000'

function apiBase() {
  return (import.meta as any).env?.VITE_API_BASE ?? DEFAULT_API
}

async function tryFetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url, { headers: { 'Accept': 'application/json' } })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return await res.json() as T
}

export async function getState(): Promise<StateResponse> {
  try {
    return await tryFetchJson<StateResponse>(`${apiBase()}/api/state`)
  } catch {
    // fallback mock (front-only mode)
    return {
      temperature_c: 28.3,
      humidity_pct: 56,
      persons_area_a: [
        { id: 1, name: '田中 太郎', status: '正常', emergency_status: '無', photo_url: null },
        { id: 2, name: '佐藤 花子', status: '検知なし', emergency_status: '転倒の可能性', photo_url: null },
      ],
      persons_area_b: [
        { id: 3, name: '鈴木 次郎', status: '正常', emergency_status: '無', photo_url: null },
        { id: 4, name: '高橋 美咲', status: '検知なし', emergency_status: '無', photo_url: null },
        { id: 5, name: '伊藤 健', status: '正常', emergency_status: '無', photo_url: null },
      ],
    }
  }
}

export async function getAllLogs(): Promise<LogItem[]> {
  try {
    return await tryFetchJson<LogItem[]>(`${apiBase()}/api/logs`)
  } catch {
    // front-only mock
    return [
      {
        id: 'evt-001',
        timestamp: new Date().toISOString(),
        person_id: 1,
        person_name: '田中 太郎',
        kind: 'FALL',
        message: '1月16日17:05 田中 太郎　田中 太郎の転倒を検知しました。',
        snapshot_url: '/assets/fallback_snapshot.png',
      },
    ]
  }
}

export async function getPersonLogs(personId: number): Promise<LogItem[]> {
  try {
    return await tryFetchJson<LogItem[]>(`${apiBase()}/api/person/${personId}/logs`)
  } catch {
    const all = await getAllLogs()
    return all.filter(l => l.person_id === personId)
  }
}
