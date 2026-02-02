import type { StateResponse, LogItem } from './types'

const DEFAULT_API = 'http://localhost:8000'
const API = (import.meta as any).env?.VITE_API_BASE ?? DEFAULT_API

export function apiBase() { return API }

export function resolveSnapshotUrl(path?: string | null) {
  if (!path) return null
  if (path.startsWith('http://') || path.startsWith('https://')) return path
  return API.replace(/\/$/, '') + path
}

async function getJson<T>(url: string): Promise<T> {
  const r = await fetch(url, { headers: { 'Accept': 'application/json' } })
  if (!r.ok) throw new Error(`HTTP ${r.status}`)
  return await r.json() as T
}

export async function getState(): Promise<StateResponse> {
  return await getJson<StateResponse>(`${API}/api/state`)
}

export async function getLogs(): Promise<LogItem[]> {
  return await getJson<LogItem[]>(`${API}/api/logs`)
}

export async function getPersonLogs(personId: number): Promise<LogItem[]> {
  return await getJson<LogItem[]>(`${API}/api/person/${personId}/logs`)
}
