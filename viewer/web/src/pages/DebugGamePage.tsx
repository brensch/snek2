import { useEffect, useMemo, useState, type CSSProperties } from 'react'
import { Link, useParams } from 'react-router-dom'
import {
  fetchDebugGame,
  type DebugGameResponse,
  type DebugMCTSNode,
  type DebugGameState,
} from '../api'

function moveName(m: number): string {
  switch (m) {
    case 0: return 'Up'
    case 1: return 'Down'
    case 2: return 'Left'
    case 3: return 'Right'
    default: return '?'
  }
}

const SNAKE_COLORS: Record<string, { body: string; head: string }> = {}
const DEFAULT_COLORS = [
  { body: '#4CAF50', head: '#2E7D32' },
  { body: '#2196F3', head: '#1565C0' },
  { body: '#FF9800', head: '#E65100' },
  { body: '#9C27B0', head: '#6A1B9A' },
]

function getSnakeColor(snakeId: string, idx: number) {
  if (!SNAKE_COLORS[snakeId]) {
    SNAKE_COLORS[snakeId] = DEFAULT_COLORS[idx % DEFAULT_COLORS.length]
  }
  return SNAKE_COLORS[snakeId]
}

function renderBoard(state: DebugGameState | undefined, snakeOrder?: string[]): string {
  if (!state) return ''
  const { width, height, food, snakes } = state
  const grid: string[][] = []
  for (let y = 0; y < height; y++) {
    grid.push(new Array(width).fill('.'))
  }
  // Food
  for (const f of food) {
    if (f.y >= 0 && f.y < height && f.x >= 0 && f.x < width) {
      grid[height - 1 - f.y][f.x] = '*'
    }
  }
  // Snakes
  snakes.forEach((snake, defaultIdx) => {
    if (!snake.alive || !snake.body?.length) return
    // Find the order index
    const orderIdx = snakeOrder?.indexOf(snake.id) ?? defaultIdx
    const letter = String.fromCharCode(65 + orderIdx) // A, B, C, D...
    snake.body.forEach((p, i) => {
      if (p.y >= 0 && p.y < height && p.x >= 0 && p.x < width) {
        grid[height - 1 - p.y][p.x] = i === 0 ? letter : letter.toLowerCase()
      }
    })
  })
  return grid.map((row) => row.join(' ')).join('\n')
}

// TreeNode component for the unified MCTS tree
// Each node represents ALL snakes making moves together (joint action)
function TreeNode({
  node,
  depth,
  snakeOrder,
  isChosen,
  onSelectNode,
}: {
  node: DebugMCTSNode
  depth: number
  snakeOrder: string[]
  isChosen: boolean
  onSelectNode?: (node: DebugMCTSNode) => void
}) {
  // Start collapsed by default
  const [expanded, setExpanded] = useState(false)
  const hasChildren = node.children && node.children.length > 0

  // Format joint moves: "A→Left, B→Right, C→Down"
  const formatJointMoves = (): { letter: string; move: string; color: string }[] | null => {
    if (!node.moves || Object.keys(node.moves).length === 0) return null
    return Object.entries(node.moves)
      .sort(([a], [b]) => a.localeCompare(b)) // Sort by snake ID
      .map(([snakeId, move]) => {
        const snakeIdx = snakeOrder.indexOf(snakeId)
        const letter = snakeIdx >= 0 ? String.fromCharCode(65 + snakeIdx) : '?'
        const color = snakeIdx >= 0 ? getSnakeColor(snakeId, snakeIdx) : { head: '#888' }
        return { letter, move: moveName(move), color: color.head }
      })
  }

  const jointMoves = formatJointMoves()

  const nodeStyle: CSSProperties = {
    padding: '4px 8px',
    margin: '2px 0',
    border: isChosen ? '2px solid #4CAF50' : '1px solid #ddd',
    borderRadius: 4,
    backgroundColor: isChosen ? '#E8F5E9' : '#fff',
    cursor: hasChildren ? 'pointer' : 'default',
    fontSize: 12,
    display: 'inline-block',
  }

  return (
    <div style={{ marginLeft: depth > 0 ? 24 : 0 }}>
      <div
        style={nodeStyle}
        onClick={() => {
          if (hasChildren) setExpanded((e) => !e)
          if (onSelectNode) onSelectNode(node)
        }}
      >
        {/* Joint moves display */}
        {jointMoves === null ? (
          <strong>Root</strong>
        ) : (
          <span>
            {jointMoves.map((m, i) => (
              <span key={i}>
                {i > 0 && ', '}
                <span style={{ fontWeight: 'bold', color: m.color }}>{m.letter}</span>
                →{m.move}
              </span>
            ))}
          </span>
        )}
        {' · '}
        N={node.n} · Q={node.q.toFixed(3)} · P={node.p.toFixed(3)} · UCB={node.ucb.toFixed(3)}
        {hasChildren && <span style={{ marginLeft: 8 }}>{expanded ? '▼' : '▶'}</span>}
      </div>

      {/* Show state if available */}
      {expanded && node.state && (
        <div style={{ marginLeft: 28, marginTop: 4, marginBottom: 4 }}>
          <pre style={{ fontSize: 11, lineHeight: 1.2, margin: 0, padding: 6, background: '#f5f5f5', borderRadius: 4, display: 'inline-block' }}>
            {renderBoard(node.state, snakeOrder)}
          </pre>
        </div>
      )}

      {expanded && hasChildren && (
        <div style={{ borderLeft: '1px solid #ccc', paddingLeft: 8 }}>
          {node.children!.map((child, i) => (
            <TreeNode
              key={i}
              node={child}
              depth={depth + 1}
              snakeOrder={snakeOrder}
              isChosen={false}
              onSelectNode={onSelectNode}
            />
          ))}
        </div>
      )}
    </div>
  )
}

// SharedTreePanel displays the unified tree (all snakes move together)
function SharedTreePanel({
  tree,
  snakeOrder,
  rootState,
  chosenPath,
}: {
  tree: DebugMCTSNode
  snakeOrder: string[]
  rootState?: DebugGameState
  chosenPath?: number[]
}) {
  const [selectedNode, setSelectedNode] = useState<DebugMCTSNode | null>(null)
  
  // Build snake legend
  const legend = snakeOrder.map((sid, i) => {
    const color = getSnakeColor(sid, i)
    const letter = String.fromCharCode(65 + i)
    return (
      <span key={sid} style={{ marginRight: 16 }}>
        <span style={{
          display: 'inline-block',
          width: 16,
          height: 16,
          backgroundColor: color.head,
          borderRadius: 4,
          marginRight: 4,
          verticalAlign: 'middle',
        }} />
        <strong style={{ color: color.head }}>{letter}</strong>
        <span style={{ fontSize: 11, color: '#666', marginLeft: 4 }}>
          {sid.slice(0, 8)}
        </span>
      </span>
    )
  })

  // Determine which state to show
  const displayState = selectedNode?.state ?? rootState

  return (
    <div style={{ border: '1px solid #ddd', padding: 12, marginBottom: 12, borderRadius: 4 }}>
      <h4 style={{ margin: '0 0 8px 0' }}>
        Unified MCTS Tree (Joint Actions)
      </h4>
      <div style={{ marginBottom: 12 }}>{legend}</div>

      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
        {/* Tree view */}
        <div style={{ flex: '1 1 600px', maxHeight: 600, overflow: 'auto' }}>
          <div style={{ fontSize: 11, color: '#666', marginBottom: 8 }}>
            Each node shows all snakes' joint action. Click to see the resulting state.
            Higher UCB = more promising to explore. P = joint prior (product of individual priors).
          </div>

          <div style={{ fontSize: 12, marginBottom: 4 }}>
            <strong>Root</strong> · N={tree.n} · Q={tree.q.toFixed(3)}
          </div>

          {tree.children?.map((child, i) => (
            <TreeNode
              key={i}
              node={child}
              depth={0}
              snakeOrder={snakeOrder}
              isChosen={false}
              onSelectNode={setSelectedNode}
            />
          ))}
        </div>

        {/* State display - shows selected node or root */}
        <div style={{ flex: '0 0 auto' }}>
          <div style={{ fontSize: 11, color: '#666', marginBottom: 4 }}>
            {selectedNode 
              ? `After joint action (turn ${displayState?.turn ?? '?'})` 
              : `Before moves (turn ${rootState?.turn ?? '?'})`}
          </div>
          <pre style={{
            fontSize: 12,
            lineHeight: 1.2,
            margin: 0,
            padding: 8,
            background: selectedNode ? '#e3f2fd' : '#fff8e1',
            borderRadius: 4,
            border: selectedNode ? '1px solid #90caf9' : '1px solid #ffe082'
          }}>
            {renderBoard(displayState, snakeOrder)}
          </pre>
          {selectedNode && (
            <button 
              onClick={() => setSelectedNode(null)}
              style={{ marginTop: 8, padding: '4px 8px', fontSize: 11 }}
            >
              Show root state
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

export default function DebugGamePage() {
  const { gameId } = useParams()
  const [game, setGame] = useState<DebugGameResponse | null>(null)
  const [error, setError] = useState<string>('')
  const [loading, setLoading] = useState<boolean>(true)
  const [turnIdx, setTurnIdx] = useState<number>(0)

  useEffect(() => {
    if (!gameId) return
    let cancelled = false
    setLoading(true)
    setGame(null)
    setTurnIdx(0)

    fetchDebugGame(gameId)
      .then((g) => {
        if (cancelled) return
        setGame(g)
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

    return () => { cancelled = true }
  }, [gameId])

  const currentTurn = game?.turns?.[turnIdx]
  const board = useMemo(() => renderBoard(currentTurn?.state, currentTurn?.snake_order), [currentTurn])

  // Determine if V2 format (has tree) or V1 format (has trees array)
  const isV2 = currentTurn?.tree !== undefined

  if (!gameId) return <div>Missing game id.</div>
  if (loading) return <div>Loading debug game…</div>
  if (error) return <div style={{ whiteSpace: 'pre-wrap' }}>Error: {error}</div>
  if (!game || !game.turns?.length) return <div>No turns found.</div>

  return (
    <div>
      <h2 style={{ marginTop: 0 }}>Debug Game</h2>
      <div style={{ marginBottom: 8 }}>
        <Link to="/">← Back to Games</Link>
      </div>

      <div style={{ marginBottom: 12, fontSize: 14 }}>
        <div><strong>Game ID:</strong> {game.game_id}</div>
        <div><strong>Model:</strong> {game.model_path}</div>
        <div><strong>Sims:</strong> {game.sims} · <strong>Cpuct:</strong> {game.cpuct}</div>
        <div><strong>Turn:</strong> {currentTurn?.turn ?? 0} ({turnIdx + 1}/{game.turn_count})</div>
        <div><strong>Format:</strong> {isV2 ? 'V2 (alternating tree)' : 'V1 (per-snake trees)'}</div>
      </div>

      {/* Turn navigation */}
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 12 }}>
        <button onClick={() => setTurnIdx((v) => Math.max(0, v - 1))} disabled={turnIdx <= 0}>
          Prev
        </button>
        <button
          onClick={() => setTurnIdx((v) => Math.min(game.turn_count - 1, v + 1))}
          disabled={turnIdx >= game.turn_count - 1}
        >
          Next
        </button>
        <input
          type="range"
          min={0}
          max={Math.max(0, game.turn_count - 1)}
          value={turnIdx}
          onChange={(e) => setTurnIdx(Number(e.target.value))}
          style={{ flex: 1, maxWidth: 400 }}
        />
      </div>

      <div style={{ display: 'flex', gap: 16, alignItems: 'flex-start', flexWrap: 'wrap' }}>
        {/* Board */}
        <div>
          <h3 style={{ margin: '0 0 8px 0' }}>Board</h3>
          <pre style={{ lineHeight: 1.2, fontSize: 14, padding: 12, border: '1px solid #ddd', margin: 0 }}>
            {board}
          </pre>
        </div>

        {/* MCTS Tree */}
        <div style={{ flex: 1, minWidth: 600 }}>
          <h3 style={{ margin: '0 0 8px 0' }}>MCTS Tree</h3>
          
          {isV2 && currentTurn?.tree ? (
            <SharedTreePanel
              tree={currentTurn.tree}
              snakeOrder={currentTurn.snake_order ?? []}
              rootState={currentTurn.state}
              chosenPath={currentTurn.chosen_path}
            />
          ) : currentTurn?.trees?.length ? (
            <div style={{ color: '#666' }}>
              V1 format (per-snake trees) - please regenerate with new debug game format
            </div>
          ) : (
            <div style={{ color: '#999' }}>No tree data for this turn</div>
          )}
        </div>
      </div>
    </div>
  )
}
