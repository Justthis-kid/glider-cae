import { useEffect } from 'react'

// Pilot convention: stick back (S / Down) = nose up, A / Left = roll left.
const BINDINGS = {
  KeyS: ['elevator', +1], ArrowDown: ['elevator', +1],
  KeyW: ['elevator', -1], ArrowUp: ['elevator', -1],
  KeyD: ['aileron', +1], ArrowRight: ['aileron', +1],
  KeyA: ['aileron', -1], ArrowLeft: ['aileron', -1],
  KeyE: ['rudder', +1],
  KeyQ: ['rudder', -1],
}

export function useKeyboardControls(setControlTarget, { onReset, onToggleRun }) {
  useEffect(() => {
    const pressed = new Set()

    const apply = () => {
      const target = { elevator: 0, aileron: 0, rudder: 0 }
      for (const code of pressed) {
        const [axis, direction] = BINDINGS[code]
        target[axis] += direction
      }
      for (const axis of Object.keys(target)) {
        setControlTarget(axis, Math.max(-1, Math.min(1, target[axis])))
      }
    }

    const onKeyDown = (event) => {
      const el = event.target
      if (el instanceof HTMLInputElement || event.metaKey || event.ctrlKey) return
      if (event.code === 'Space' && el instanceof HTMLButtonElement) return // native button press
      if (event.code === 'KeyR' && !event.repeat) onReset()
      else if (event.code === 'Space') {
        event.preventDefault()
        if (!event.repeat) onToggleRun()
      } else if (event.code in BINDINGS) {
        event.preventDefault()
        pressed.add(event.code)
        apply()
      }
    }
    const onKeyUp = (event) => {
      if (pressed.delete(event.code)) apply()
    }
    const onBlur = () => {
      pressed.clear()
      apply()
    }

    window.addEventListener('keydown', onKeyDown)
    window.addEventListener('keyup', onKeyUp)
    window.addEventListener('blur', onBlur)
    return () => {
      window.removeEventListener('keydown', onKeyDown)
      window.removeEventListener('keyup', onKeyUp)
      window.removeEventListener('blur', onBlur)
    }
  }, [setControlTarget, onReset, onToggleRun])
}
