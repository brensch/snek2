import { useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { fetchGameTurns, type Turn } from '../api'
import { renderAsciiBoard } from '../board'

export default function GamePage() {
  const { gameId } = useParams()
  const [turns, setTurns] = useState<Turn[]>([])
  const [error, setError] = useState<string>('')
  const [loading, setLoading] = useState<boolean>(true)
  const [idx, setIdx] = useState<number>(0)
  const [playing, setPlaying] = useState<boolean>(false)
  const [speedMs, setSpeedMs] = useState<number>(50)

  useEffect(() => {
    if (!gameId) return
    let cancelled = false
    setLoading(true)
    setTurns([])
    setIdx(0)
    setPlaying(false)

    fetchGameTurns(gameId)
      .then((t) => {
        if (cancelled) return
        setTurns(t)
        setError('')
      })
      .catch((e: unknown) => {
        if (cancelled) return
        setError(e instanceof Error ? e.message : String(e))
      })
      .finally(() => {
        if (cancelled) return
        setLoading(false)
      })

    return () => {
      cancelled = true
    }
  }, [gameId])

  useEffect(() => {
    if (!playing) return
    if (turns.length === 0) return

    const t = window.setInterval(() => {
      setIdx((cur) => {
        const next = cur + 1
        if (next >= turns.length) {
          // Auto-pause at the end.
          setPlaying(false)
          return cur
        }
        return next
      })
    }, speedMs)

    return () => {
      window.clearInterval(t)
    }
  }, [playing, speedMs, turns.length])

  const current = turns[Math.min(idx, Math.max(0, turns.length - 1))]
  const board = useMemo(() => (current ? renderAsciiBoard(current) : ''), [current])

  if (!gameId) return <div>Missing game id.</div>
  if (loading) return <div>Loading…</div>
  if (error) return <div style={{ whiteSpace: 'pre-wrap' }}>Error: {error}</div>
  if (turns.length === 0) return <div>No turns found for this game.</div>

  return (
    <div>
      <h2 style={{ marginTop: 0 }}>Game</h2>
      <div style={{ marginBottom: 8 }}>
        <Link to="/">← Back</Link>
      </div>

      <div style={{ marginBottom: 12 }}>
        <div>
          <strong>ID:</strong> {gameId}
        </div>
        <div>
          <strong>Turn:</strong> {current.turn} ({idx + 1}/{turns.length})
        </div>
      </div>

      <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 12 }}>
        <button onClick={() => setIdx((v) => Math.max(0, v - 1))} disabled={idx <= 0}>
          Prev
        </button>
        <button onClick={() => setPlaying((p) => !p)} disabled={idx >= turns.length - 1}>
          {playing ? 'Pause' : 'Play'}
        </button>
        <button
          onClick={() => setIdx((v) => Math.min(turns.length - 1, v + 1))}
          disabled={idx >= turns.length - 1}
        >
          Next
        </button>

        <label style={{ marginLeft: 12 }}>
          Speed:{' '}
          <select value={speedMs} onChange={(e) => setSpeedMs(Number(e.target.value))}>
            <option value={50}>50ms</option>
            <option value={100}>100ms</option>
            <option value={200}>200ms</option>
            <option value={500}>500ms</option>
            <option value={1000}>1000ms</option>
          </select>
        </label>
      </div>

      <div style={{ marginBottom: 12 }}>
        <input
          type="range"
          min={0}
          max={Math.max(0, turns.length - 1)}
          value={idx}
          onChange={(e) => setIdx(Number(e.target.value))}
          style={{ width: '100%' }}
        />
      </div>

      <pre style={{ lineHeight: 1.2, fontSize: 14, padding: 12, border: '1px solid #ddd', overflowX: 'auto' }}>
        {board}
      </pre>

      <div style={{ marginTop: 12 }}>
        <h3 style={{ margin: '8px 0' }}>Snakes</h3>
        <ul>
          {current.snakes.map((s) => (
            <li key={s.id}>
              {s.id} — alive: {String(s.alive)} — health: {s.health} — value: {s.value}
            </li>
          ))}
        </ul>
      </div>
    </div>
  )
}
