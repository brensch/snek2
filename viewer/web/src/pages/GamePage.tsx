import { useEffect, useMemo, useState, type CSSProperties } from 'react'
import { Link, useParams } from 'react-router-dom'
import {
  fetchGameTurns,
  fetchMctsAll,
  simulateTurn,
  type MctsAllResponse,
  type MctsMove,
  type MctsNode,
  type Snake,
  type Turn,
} from '../api'
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

function fmtMoveName(m: number): string {
  switch (m) {
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

function MctsCross({
  node,
  onPick,
}: {
  node: MctsNode
  onPick: (move: number) => void
}) {
  const cell: CSSProperties = {
    width: 92,
    height: 62,
    border: '1px solid #ddd',
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'center',
    alignItems: 'center',
    fontSize: 12,
    lineHeight: 1.1,
    cursor: 'pointer',
    userSelect: 'none',
  }

  const center: CSSProperties = {
    ...cell,
    cursor: 'default',
    color: '#666',
    fontSize: 12,
  }

  const render = (mv: MctsMove, label: string) => {
    const disabled = !mv.exists
    return (
      <div
        style={{
          ...cell,
          cursor: disabled ? 'not-allowed' : 'pointer',
          opacity: mv.exists ? 1 : 0.35,
          fontWeight: mv.exists ? 600 : 400,
        }}
        title={mv.exists ? `${fmtMoveName(mv.move)}\nUCB=${mv.ucb.toFixed(3)} N=${mv.n} Q=${mv.q.toFixed(3)} P=${mv.p.toFixed(3)}` : 'Illegal/unexpanded'}
        onClick={() => {
          if (disabled) return
          onPick(mv.move)
        }}
      >
        <div>{label}</div>
        <div style={{ color: '#444' }}>UCB {mv.ucb.toFixed(3)}</div>
        <div style={{ color: '#666' }}>N {mv.n} · Q {mv.q.toFixed(3)} · P {mv.p.toFixed(3)}</div>
      </div>
    )
  }

  const up = node.moves[0]
  const down = node.moves[1]
  const left = node.moves[2]
  const right = node.moves[3]

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '92px 92px 92px', gap: 6 }}>
      <div />
      {render(up, 'U')}
      <div />

      {render(left, 'L')}
      <div style={center}>
        <div>visits {node.visit_count}</div>
        <div>V {node.value.toFixed(3)}</div>
      </div>
      {render(right, 'R')}

      <div />
      {render(down, 'D')}
      <div />
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

  const [mctsAll, setMctsAll] = useState<MctsAllResponse | null>(null)
  const [mctsMoves, setMctsMoves] = useState<Record<string, number>>({})
  const [mctsLoading, setMctsLoading] = useState<boolean>(false)
  const [mctsError, setMctsError] = useState<string>('')
  const [simTurn, setSimTurn] = useState<Turn | null>(null)
  const [simError, setSimError] = useState<string>('')

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

  const simBoard = useMemo(() => (simTurn ? renderAsciiBoard(simTurn) : ''), [simTurn])

  useEffect(() => {
    // Reset MCTS panels when stepping turns.
    setMctsAll(null)
    setMctsMoves({})
    setMctsError('')
    setSimTurn(null)
    setSimError('')
  }, [idx, gameId])

  useEffect(() => {
    if (!mctsAll) return
    if (!mctsAll.state) return
    if (!mctsAll.snakes?.length) return

    let cancelled = false
    setSimError('')
    simulateTurn(mctsAll.state, mctsMoves)
      .then((t) => {
        if (cancelled) return
        setSimTurn(t)
      })
      .catch((e: unknown) => {
        if (cancelled) return
        setSimTurn(null)
        setSimError(e instanceof Error ? e.message : String(e))
      })

    return () => {
      cancelled = true
    }
  }, [mctsAll, mctsMoves])

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
          <h3 style={{ margin: '0 0 8px 0' }}>MCTS explorer</h3>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap', marginBottom: 8 }}>
            <button
              disabled={mctsLoading}
              onClick={() => {
                if (!gameId) return
                setMctsLoading(true)
                setMctsError('')
                setMctsAll(null)
                setMctsMoves({})
                setSimTurn(null)
                setSimError('')
                fetchMctsAll(gameId, current.turn, 100, 3, 1.0)
                  .then((r) => {
                    setMctsAll(r)
                    const moves: Record<string, number> = {}
                    for (const st of r.snakes ?? []) {
                      if (!st.snake_id) continue
                      if (typeof st.best_move === 'number' && st.best_move >= 0) {
                        moves[st.snake_id] = st.best_move
                      } else {
                        moves[st.snake_id] = 0
                      }
                    }
                    setMctsMoves(moves)
                  })
                  .catch((e: unknown) => {
                    setMctsError(e instanceof Error ? e.message : String(e))
                  })
                  .finally(() => setMctsLoading(false))
              }}
            >
              {mctsLoading ? 'Running…' : 'Re-run MCTS (100)'}
            </button>
          </div>

          {mctsError ? <div style={{ whiteSpace: 'pre-wrap', marginBottom: 8 }}>Error: {mctsError}</div> : null}

          {mctsAll ? (
            <div style={{ marginBottom: 16 }}>
              <div style={{ color: '#666', fontSize: 12, marginBottom: 6 }}>
                sims: {mctsAll.sims} · cpuct: {mctsAll.cpuct} · depth: {mctsAll.depth}
              </div>
        <div style={{ color: '#666', fontSize: 12, marginBottom: 10 }}>
        MCTS explorer re-runs MCTS using the current ONNX model. The “Snakes” section below shows the recorded move/probs
        stored in the parquet for this game (which may have been generated with a different model).
        </div>

              {(mctsAll.snakes ?? []).map((s) => {
                const { head } = snakeLetters(s.snake_idx)
                const chosen = mctsMoves[s.snake_id]
				const recorded = current.snakes[s.snake_idx]
				const recordedMove = recorded && typeof recorded.policy === 'number' && recorded.policy >= 0 ? fmtMoveName(recorded.policy) : '—'
                return (
                  <div key={s.snake_id} style={{ marginBottom: 12 }}>
                    <div style={{ marginBottom: 6, fontSize: 12 }}>
                      <strong>{head}</strong> — {s.snake_id}
                      {typeof chosen === 'number' ? ` · chosen: ${fmtMoveName(chosen)}` : ''}
                      {typeof s.best_move === 'number' && s.best_move >= 0 ? ` · best: ${fmtMoveName(s.best_move)}` : ''}
					  {` · recorded: ${recordedMove}`}
                    </div>
                    {s.root ? (
                      <MctsCross
                        node={s.root}
                        onPick={(move) => {
                          setMctsMoves((m) => ({ ...m, [s.snake_id]: move }))
                        }}
                      />
                    ) : (
                      <div style={{ fontSize: 12, color: '#666' }}>dead / no root</div>
                    )}
                  </div>
                )
              })}

              {simError ? <div style={{ whiteSpace: 'pre-wrap', marginTop: 8 }}>Sim error: {simError}</div> : null}
              {simBoard ? (
                <pre
                  style={{
                    lineHeight: 1.2,
                    fontSize: 14,
                    padding: 12,
                    border: '1px solid #ddd',
                    overflowX: 'auto',
                    margin: '10px 0 0 0',
                  }}
                >
                  {simBoard}
                </pre>
              ) : null}
            </div>
          ) : null}

          <h3 style={{ margin: '12px 0 8px 0' }}>Snakes</h3>
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
