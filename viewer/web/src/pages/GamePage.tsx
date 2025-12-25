import { useEffect, useMemo, useState, type CSSProperties } from 'react'
import { Link, useParams } from 'react-router-dom'
import { fetchGameTurns, type Snake, type Turn } from '../api'
import { renderAsciiBoard, snakeLetters } from '../board'

function clamp01(x: number): number {
  if (!Number.isFinite(x)) return 0
  if (x < 0) return 0
  if (x > 1) return 1
  return x
}

function policyName(p: number): string {
  switch (p) {
    case 0:
      return 'Up'
    case 1:
      return 'Down'
    case 2:
      return 'Left'
    case 3:
      return 'Right'
    default:
      return 'Unknown'
  }
}

function policyProbs4(s: Snake): [number, number, number, number] | null {
  const pp = s.policy_probs
  if (pp && pp.length === 4 && pp.every((x) => Number.isFinite(x))) {
    return [pp[0], pp[1], pp[2], pp[3]]
  }
  if (Number.isFinite(s.policy) && s.policy >= 0 && s.policy < 4) {
    const out: [number, number, number, number] = [0, 0, 0, 0]
    out[s.policy] = 1
    return out
  }
  return null
}

function PolicyCross({ snake }: { snake: Snake }) {
  const probs = policyProbs4(snake)
  const chosen = Number.isFinite(snake.policy) ? snake.policy : -1

  const cell: CSSProperties = {
    width: 72,
    height: 44,
    border: '1px solid #ddd',
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'center',
    alignItems: 'center',
    fontSize: 12,
    lineHeight: 1.1,
  }

  const center: CSSProperties = {
    ...cell,
    fontSize: 14,
    color: '#666',
  }

  const entry = (dir: 0 | 1 | 2 | 3, label: string) => {
    const p = probs ? probs[dir] : null
    const isChosen = chosen === dir
    return (
      <div style={{ ...cell, fontWeight: isChosen ? 700 : 400 }}>
        <div>{isChosen ? `*${label}` : label}</div>
        <div style={{ color: '#666' }}>{p == null ? '—' : `${Math.round(clamp01(p) * 100)}%`}</div>
      </div>
    )
  }

  return (
    <div>
      <div style={{ display: 'grid', gridTemplateColumns: '72px 72px 72px', gap: 6 }}>
        <div />
        {entry(0, 'U')}
        <div />

        {entry(2, 'L')}
        <div style={center}>•</div>
        {entry(3, 'R')}

        <div />
        {entry(1, 'D')}
        <div />
      </div>
      <div style={{ marginTop: 6, fontSize: 12, color: '#666' }}>
        policy: {snake.policy} ({policyName(snake.policy)})
      </div>
    </div>
  )
}

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

      <div style={{ display: 'flex', gap: 16, alignItems: 'flex-start', flexWrap: 'wrap' }}>
        <pre style={{ lineHeight: 1.2, fontSize: 14, padding: 12, border: '1px solid #ddd', overflowX: 'auto', margin: 0 }}>
          {board}
        </pre>

        <div style={{ minWidth: 260 }}>
          <h3 style={{ margin: '0 0 8px 0' }}>Snakes</h3>
          {current.snakes.map((s, i) => {
            const { head } = snakeLetters(i)
            return (
              <div key={s.id} style={{ marginBottom: 16 }}>
                <div style={{ marginBottom: 6 }}>
                  <strong>{head}</strong> — id: {s.id} — alive: {String(s.alive)} — health: {s.health} — value: {s.value}
                </div>
                <PolicyCross snake={s} />
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
