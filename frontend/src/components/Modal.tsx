import React, { useEffect } from 'react'

export default function Modal(props: { title: string; open: boolean; onClose: () => void; children: React.ReactNode }) {
  const { title, open, onClose, children } = props

  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onClose])

  if (!open) return null

  return (
    <div className="modalOverlay" role="dialog" aria-modal="true" onMouseDown={onClose}>
      <div className="modal" onMouseDown={(e) => e.stopPropagation()}>
        <div className="modalHeader">
          <div className="modalTitle">{title}</div>
          <button className="btn" onClick={onClose}>閉じる</button>
        </div>
        <div className="modalBody">{children}</div>
      </div>
    </div>
  )
}
