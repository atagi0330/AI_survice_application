export type Person = {
  id: number
  name: string
  photo_url?: string | null
  status: string
  emergency_status: string
}

export type LogItem = {
  id: string
  timestamp: string
  person_id: number
  person_name: string
  message: string
  snapshot_url?: string | null
  kind: string // FALL, FOCUS, INFO ... (拡張可)
}

export type StateResponse = {
  temperature_c: number
  humidity_pct: number
  persons_area_a: Person[]
  persons_area_b: Person[]
}
