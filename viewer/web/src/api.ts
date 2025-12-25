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

export type Turn = {
  game_id: string
  turn: number
  width: number
  height: number
  food: Point[]
  hazards: Point[]
  snakes: Snake[]
  source: string
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
