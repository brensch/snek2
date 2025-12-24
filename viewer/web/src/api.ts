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
}

export type GamesResponse = {
  total: number
  games: GameSummary[]
}

export type Point = { x: number; y: number }

export type Snake = {
  id: string
  alive: boolean
  health: number
  body: Point[]
  policy: number
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
