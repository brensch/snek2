import type { Turn } from './api'

function snakeLetters(i: number) {
  const head = String.fromCharCode('A'.charCodeAt(0) + (i % 26))
  const body = String.fromCharCode('a'.charCodeAt(0) + (i % 26))
  return { head, body }
}

export function renderAsciiBoard(t: Turn): string {
  const w = t.width
  const h = t.height
  if (w <= 0 || h <= 0) return ''

  const grid: string[][] = Array.from({ length: h }, () => Array.from({ length: w }, () => '.'))

  for (const p of t.hazards ?? []) {
    if (p.x >= 0 && p.x < w && p.y >= 0 && p.y < h) grid[p.y][p.x] = 'X'
  }
  for (const p of t.food ?? []) {
    if (p.x >= 0 && p.x < w && p.y >= 0 && p.y < h) grid[p.y][p.x] = 'F'
  }

  ;(t.snakes ?? []).forEach((s, idx) => {
    const { head, body } = snakeLetters(idx)
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
