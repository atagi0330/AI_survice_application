import React, { useEffect } from 'react'

type Props = {
  title: string
  isOpen: boolean
  onClose: () => void
  children: React.ReactNode
}

export default function Modal({ title, isOpen, onClose, children }: Props) {
  useEffect(() => {
    if (!isOpen) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [isOpen, onClose])

  if (!isOpen) return null

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
