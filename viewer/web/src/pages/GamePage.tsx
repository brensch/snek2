import { useEffect, useMemo, useState, type CSSProperties } from 'react'
import { Link, useParams } from 'react-router-dom'
import {
  fetchGameTurns,
  runInference,
  type Snake,
  type Turn,
  type RunInferenceResponse,
  type DebugMCTSNode,
} from '../api'
import { renderAsciiBoard, snakeLetters } from '../board'

const simsPerMove = 500 // Number of MCTS sims per move

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

// Render a DebugGameState as ASCII board
function renderNodeState(
  state: { width: number; height: number; food: { x: number; y: number }[]; snakes: { id: string; body: { x: number; y: number }[] }[] },
  originalSnakes: { id: string }[]
): string {
  const w = state.width
  const h = state.height
  if (w <= 0 || h <= 0) return ''

  const grid: string[][] = Array.from({ length: h }, () => Array.from({ length: w }, () => '.'))

  for (const p of state.food ?? []) {
    if (p.x >= 0 && p.x < w && p.y >= 0 && p.y < h) grid[p.y][p.x] = 'F'
  }

  ;(state.snakes ?? []).forEach((s) => {
    // Find the index in the original snakes array for consistent lettering
    const origIdx = originalSnakes.findIndex((os) => os.id === s.id)
    const head = origIdx >= 0 ? String.fromCharCode('A'.charCodeAt(0) + origIdx) : '?'
    const body = origIdx >= 0 ? String.fromCharCode('a'.charCodeAt(0) + origIdx) : '?'
    ;(s.body ?? []).forEach((p, j) => {
      if (p.x < 0 || p.x >= w || p.y < 0 || p.y >= h) return
      grid[p.y][p.x] = j === 0 ? head : body
    })
  })

  // Render with y=max at top (Battlesnake coords are bottom-left).
  const lines: string[] = []
  for (let y = h - 1; y >= 0; y--) {
    lines.push(grid[y].join(' '))
  }
  return lines.join('\n')
}

// Snake colors for visualization
const SNAKE_COLORS = [
  '#4CAF50', // A - green
  '#2196F3', // B - blue
  '#F44336', // C - red
  '#FF9800', // D - orange
  '#9C27B0', // E - purple
  '#00BCD4', // F - cyan
]

// Move names
const MOVE_NAMES = ['Up', 'Down', 'Left', 'Right']

// Alternating tree node component - shows single snake moves (each level is one snake's turn)
function AlternatingTreeNode({
  node,
  depth,
  turnSnakes,
  selectedNode,
  onSelectNode,
}: {
  node: DebugMCTSNode
  depth: number
  turnSnakes: { id: string }[] // Original snakes from the turn (to get correct letter)
  selectedNode: DebugMCTSNode | null
  onSelectNode: (n: DebugMCTSNode) => void
}) {
  const [expanded, setExpanded] = useState(false) // Start collapsed
  const hasChildren = node.children && node.children.length > 0

  // Get the snake who MADE the move to reach this node (from the moves field)
  // This is different from snake_id which is the snake whose turn it is NEXT
  const moveSnakeId = node.moves ? Object.keys(node.moves)[0] : ''
  const originalIdx = moveSnakeId ? turnSnakes.findIndex((s) => s.id === moveSnakeId) : -1
  const letter = originalIdx >= 0 && originalIdx < 26 ? String.fromCharCode(65 + originalIdx) : '?'
  const color = originalIdx >= 0 ? SNAKE_COLORS[originalIdx % SNAKE_COLORS.length] : '#999'

  // Get the move made to reach this node
  const getMove = (): string => {
    if (!node.moves || Object.keys(node.moves).length === 0) return ''
    // For alternating tree, there's only one entry
    const moveVal = Object.values(node.moves)[0]
    return MOVE_NAMES[moveVal] ?? '?'
  }

  const isRoot = depth === 0
  const move = getMove()

  // Show if this is the start of a new round (all snakes have moved)
  const isNewRound = node.snake_index === 0 && depth > 0
  const isSelected = selectedNode === node

  const nodeStyle: CSSProperties = {
    padding: '4px 8px',
    margin: '2px 0',
    border: `2px solid ${isSelected ? '#000' : color}`,
    borderRadius: 4,
    backgroundColor: isNewRound ? '#f0f0f0' : isSelected ? '#ffffcc' : '#fff',
    cursor: 'pointer',
    fontSize: 12,
    display: 'inline-block',
  }

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation()
    // Always select this node to show its state
    onSelectNode(node)
    // Also toggle expand/collapse if it has children
    if (hasChildren) {
      setExpanded((exp) => !exp)
    }
  }

  return (
    <div style={{ marginLeft: depth > 0 ? 16 : 0 }}>
      <div
        style={nodeStyle}
        onClick={handleClick}
        title={`Snake ${letter} (${moveSnakeId.slice(0, 8)}...)`}
      >
        {isRoot ? (
          <strong>Root</strong>
        ) : (
          <>
            <span style={{ color, fontWeight: 'bold' }}>{letter}</span>
            →{move}
          </>
        )}
        {' · '}
        <span style={{ color: '#666' }}>
          N={node.n} Q={node.q.toFixed(2)} P={node.p.toFixed(2)} UCB={node.ucb.toFixed(2)}
        </span>
        {isNewRound && <span style={{ marginLeft: 6, color: '#999', fontSize: 10 }}>↻ new round</span>}
        {hasChildren && <span style={{ marginLeft: 8 }}>{expanded ? '▼' : '▶'}</span>}
      </div>
      {expanded && hasChildren && (
        <div style={{ borderLeft: `2px solid ${color}`, paddingLeft: 8 }}>
          {node.children!.map((child, i) => (
            <AlternatingTreeNode
              key={i}
              node={child}
              depth={depth + 1}
              turnSnakes={turnSnakes}
              selectedNode={selectedNode}
              onSelectNode={onSelectNode}
            />
          ))}
        </div>
      )}
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

  // Unified tree inference state
  const [unifiedTree, setUnifiedTree] = useState<RunInferenceResponse | null>(null)
  const [unifiedLoading, setUnifiedLoading] = useState<boolean>(false)
  const [unifiedError, setUnifiedError] = useState<string>('')
  const [selectedTreeNode, setSelectedTreeNode] = useState<DebugMCTSNode | null>(null)

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

  useEffect(() => {
    // Reset unified tree when stepping turns.
    setUnifiedTree(null)
    setUnifiedError('')
    setSelectedTreeNode(null)
  }, [idx, gameId])

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
        <div>
          {/* Current Turn Board */}
          <div style={{ marginBottom: 8, fontSize: 12, color: '#666' }}>Turn {current.turn}</div>
          <pre style={{ lineHeight: 1.2, fontSize: 14, padding: 12, border: '1px solid #ddd', overflowX: 'auto', margin: 0 }}>
            {board}
          </pre>

          {/* Selected node state visualization - below main board */}
          {selectedTreeNode?.state ? (
            <div style={{ marginTop: 16 }}>
              <div style={{ marginBottom: 8, fontSize: 12, color: '#666' }}>
                Explored State (Turn {selectedTreeNode.state.turn})
              </div>
              <pre style={{ lineHeight: 1.2, fontSize: 14, padding: 12, border: '1px solid #ddd', overflowX: 'auto', margin: 0 }}>
                {renderNodeState(selectedTreeNode.state, current.snakes)}
              </pre>
            </div>
          ) : selectedTreeNode ? (
            <div style={{ marginTop: 16, padding: 12, border: '1px solid #ddd', color: '#999', fontSize: 12 }}>
              Selected node has no state (intermediate move in round)
            </div>
          ) : null}
        </div>

        <div style={{ minWidth: 260 }}>
          {/* Model Path (if available) */}
          {current.model_path ? (
            <div style={{ marginBottom: 12, fontSize: 12, color: '#666' }}>
              <strong>Model:</strong> {current.model_path.split('/').pop()}
              <div style={{ fontSize: 10, color: '#999', wordBreak: 'break-all' }}>{current.model_path}</div>
            </div>
          ) : null}

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

          {/* Unified Tree Section */}
          <h3 style={{ margin: '24px 0 8px 0' }}>Alternating MCTS Tree</h3>
          <div style={{ color: '#666', fontSize: 12, marginBottom: 8 }}>
            Runs MCTS where each snake takes turns (alternating). Click to expand, Shift+click to view state.
          </div>
          <button
            style={{ marginBottom: 12 }}
            disabled={unifiedLoading}
            onClick={() => {
              if (!gameId) return
              setUnifiedLoading(true)
              setUnifiedError('')
              setSelectedTreeNode(null)
              runInference({ game_id: gameId, turn: current.turn, sims: simsPerMove })
                .then((res) => {
                  setUnifiedTree(res)
                })
                .catch((e: unknown) => {
                  setUnifiedError(e instanceof Error ? e.message : String(e))
                })
                .finally(() => setUnifiedLoading(false))
            }}
          >
            {unifiedLoading ? 'Running…' : `Expand Alternating Tree (${simsPerMove} sims)`}
          </button>

          {unifiedError ? <div style={{ color: 'red', marginBottom: 8 }}>Error: {unifiedError}</div> : null}

          {unifiedTree ? (
            <div style={{ marginBottom: 16 }}>
              <div style={{ color: '#666', fontSize: 12, marginBottom: 6 }}>
                sims: {unifiedTree.sims} · snakes: {current.snakes.map((s, i) => `${String.fromCharCode(65 + i)}=${s.id.slice(0, 8)}`).join(', ')}
              </div>
              <AlternatingTreeNode
                node={unifiedTree.tree}
                depth={0}
                turnSnakes={current.snakes}
                selectedNode={selectedTreeNode}
                onSelectNode={setSelectedTreeNode}
              />
            </div>
          ) : null}
        </div>
      </div>
    </div>
  )
}
