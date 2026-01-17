export type GameSummary = {
  game_id: string
  started_ns?: number | null
  min_turn: number
  max_turn: number
  turn_count: number
  width: number
  height: number
  source: string
  file: string
	results?: string
}

export type GamesResponse = {
  total: number
  games: GameSummary[]
}

export type StatsPoint = {
  t_ns: number

  selfplay_games: number
  selfplay_total_turns: number
  selfplay_wins: number
  selfplay_draws: number

  scraped_games: number
  scraped_total_turns: number
  scraped_wins: number
  scraped_draws: number
}

export type StatsResponse = {
  from_ns: number
  to_ns: number
  bucket_ns: number
  points: StatsPoint[]
}

export type Point = { x: number; y: number }

export type Snake = {
  id: string
  alive: boolean
  health: number
  body: Point[]
  policy: number
  policy_probs?: number[]
  value: number
}

// JointChildSummary represents a joint action explored at the MCTS root
export type JointChildSummary = {
  moves: Record<string, number>  // snakeID -> move
  n: number                      // visit count
  value_sum: number
  q: number                      // average value
  p: number                      // joint prior
}

export type Turn = {
  game_id: string
  turn: number
  width: number
  height: number
  food: Point[]
  hazards: Point[]
  snakes: Snake[]
  source: string
  // Model path (resolved, present in selfplay games)
  model_path?: string
  // MCTS root children summary (present in selfplay games)
  mcts_root?: JointChildSummary[]
}

export type MctsMove = {
  move: number
  exists: boolean
  n: number
  q: number
  p: number
  ucb: number
  child?: MctsNode
}

export type MctsNode = {
  visit_count: number
  value: number
  state?: Turn
  moves: [MctsMove, MctsMove, MctsMove, MctsMove]
}

export type MctsResponse = {
  game_id: string
  turn: number
  ego_idx: number
  ego_id: string
  sims: number
  cpuct: number
  depth: number
  max_depth: number
  best_move: number
  root: MctsNode
}

export type MctsSnakeTree = {
  snake_idx: number
  snake_id: string
  best_move: number
  root?: MctsNode | null
}

export type MctsAllResponse = {
  game_id: string
  turn: number
  sims: number
  cpuct: number
  depth: number
  state: Turn
  snakes: MctsSnakeTree[]
}

export async function fetchGames(
  limit = 200,
  offset = 0,
  sort: string = 'started_ns',
  dir: 'asc' | 'desc' = 'desc',
): Promise<GamesResponse> {
  const res = await fetch(`/api/games?limit=${limit}&offset=${offset}&sort=${encodeURIComponent(sort)}&dir=${dir}`)
  if (!res.ok) throw new Error(await res.text())
  return (await res.json()) as GamesResponse
}

export async function fetchGameTurns(gameId: string): Promise<Turn[]> {
  const res = await fetch(`/api/games/${encodeURIComponent(gameId)}/turns`)
  if (!res.ok) throw new Error(await res.text())
  return (await res.json()) as Turn[]
}

export async function fetchStats(fromNs: number, toNs: number, bucketNs: number): Promise<StatsResponse> {
  const url = `/api/stats?from_ns=${fromNs}&to_ns=${toNs}&bucket_ns=${bucketNs}`
  const res = await fetch(url)
  if (!res.ok) throw new Error(await res.text())
  return (await res.json()) as StatsResponse
}

export async function fetchMcts(
  gameId: string,
  turn: number,
  egoIdx: number,
  sims: number = 100,
  depth: number = 3,
  cpuct: number = 1.0,
): Promise<MctsResponse> {
  const url = `/api/mcts?game_id=${encodeURIComponent(gameId)}&turn=${turn}&ego_idx=${egoIdx}&sims=${sims}&depth=${depth}&cpuct=${cpuct}`
  const res = await fetch(url)
  if (!res.ok) throw new Error(await res.text())
  return (await res.json()) as MctsResponse
}

export async function fetchMctsAll(
  gameId: string,
  turn: number,
  sims: number = 100,
  depth: number = 3,
  cpuct: number = 1.0,
): Promise<MctsAllResponse> {
  const url = `/api/mcts_all?game_id=${encodeURIComponent(gameId)}&turn=${turn}&sims=${sims}&depth=${depth}&cpuct=${cpuct}`
  const res = await fetch(url)
  if (!res.ok) throw new Error(await res.text())
  return (await res.json()) as MctsAllResponse
}

export async function simulateTurn(state: Turn, moves: Record<string, number>): Promise<Turn> {
  const res = await fetch('/api/simulate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ state, moves }),
  })
  if (!res.ok) throw new Error(await res.text())
  return (await res.json()) as Turn
}

// Debug game types

// Alternating MCTS node - each node represents ONE snake's move decision
export type DebugMCTSNode = {
  // Moves - for alternating tree, this is just {snakeID: move}
  moves?: Record<string, number>
  
  n: number
  value_sum: number
  q: number
  p: number  // Prior probability for this move
  ucb: number
  
  // SnakeID is the snake whose turn it is at this node
  snake_id?: string
  
  // SnakeIndex is the position in the snake order (0, 1, 2, ...)
  snake_index: number
  
  // State is only set at the start of each round (when all snakes have moved)
  state?: DebugGameState
  
  // Per-snake priors at this node
  snake_priors?: Record<string, number[]>
  
  children?: DebugMCTSNode[]
  
  // Legacy fields for old format compatibility
  move?: number
  is_resolved?: boolean
  pending_moves?: Record<string, number>
  next_snake?: string
}

export type DebugGameState = {
  turn: number
  width: number
  height: number
  you_id: string
  food: Point[]
  snakes: DebugSnakeState[]
}

export type DebugSnakeState = {
  id: string
  health: number
  body: Point[]
  alive: boolean
}

export type DebugSnakeTree = {
  snake_id: string
  snake_idx: number
  best_move: number
  root?: DebugMCTSNode
}

export type DebugTurnData = {
  game_id: string
  model_path: string
  turn: number
  sims: number
  cpuct: number
  state?: DebugGameState
  // V1: Separate tree per snake (deprecated)
  trees?: DebugSnakeTree[]
  // V2: Single shared tree that alternates between snakes
  snake_order?: string[]
  tree?: DebugMCTSNode
  chosen_path?: number[]
}

export type DebugGameSummary = {
  game_id: string
  model_path: string
  turn_count: number
  sims: number
  cpuct: number
  file_name: string
}

export type DebugGameResponse = {
  game_id: string
  model_path: string
  turn_count: number
  sims: number
  cpuct: number
  turns: DebugTurnData[]
}

export async function fetchDebugGames(): Promise<DebugGameSummary[]> {
  const res = await fetch('/api/debug_games')
  if (!res.ok) throw new Error(await res.text())
  return (await res.json()) as DebugGameSummary[]
}

export async function fetchDebugGame(gameId: string): Promise<DebugGameResponse> {
  const res = await fetch(`/api/debug_games/${encodeURIComponent(gameId)}`)
  if (!res.ok) throw new Error(await res.text())
  return (await res.json()) as DebugGameResponse
}

// Run inference request/response types
export type RunInferenceRequest = {
  game_id: string
  turn: number
  sims: number
}

export type RunInferenceResponse = {
  game_id: string
  turn: number
  sims: number
  snake_order: string[]
  state: DebugGameState
  tree: DebugMCTSNode
}

export async function runInference(req: RunInferenceRequest): Promise<RunInferenceResponse> {
  const res = await fetch('/api/run_inference', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })
  if (!res.ok) throw new Error(await res.text())
  return (await res.json()) as RunInferenceResponse
}
