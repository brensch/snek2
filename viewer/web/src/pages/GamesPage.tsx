import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { fetchGames, type GameSummary } from '../api'

type SortKey = 'started_ns' | 'game_id' | 'turn_count' | 'width' | 'height' | 'source' | 'file'
type SortDir = 'asc' | 'desc'

function formatLocalTimeFromNs(ns?: number | null): string {
  if (ns == null) return ''
  const ms = Math.floor(ns / 1_000_000)
  const d = new Date(ms)
  if (Number.isNaN(d.getTime())) return ''
  return d.toLocaleString()
}

export default function GamesPage() {
  const [games, setGames] = useState<GameSummary[]>([])
  const [total, setTotal] = useState<number>(0)
  const [error, setError] = useState<string>('')
  const [loading, setLoading] = useState<boolean>(true)
  const [pageSize, setPageSize] = useState<number>(50)
  const [page, setPage] = useState<number>(0)
  const [sortKey, setSortKey] = useState<SortKey>('started_ns')
  const [sortDir, setSortDir] = useState<SortDir>('desc')

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    fetchGames(pageSize, page * pageSize, sortKey, sortDir)
      .then((resp) => {
        if (cancelled) return
        setGames(resp.games)
        setTotal(resp.total)
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
  }, [page, pageSize, sortDir, sortKey])

  if (loading) return <div>Loading…</div>
  if (error) return <div style={{ whiteSpace: 'pre-wrap' }}>Error: {error}</div>

  const totalPages = Math.max(1, Math.ceil(total / pageSize))
  const clampedPage = Math.min(page, totalPages - 1)
  if (clampedPage !== page) setPage(clampedPage)

  const toggleSort = (key: SortKey) => {
    if (key === sortKey) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'))
      setPage(0)
      return
    }
    setSortKey(key)
    setSortDir(key === 'started_ns' ? 'desc' : 'asc')
    setPage(0)
  }

  const sortLabel = (key: SortKey) => {
    if (key !== sortKey) return ''
    return sortDir === 'asc' ? ' ▲' : ' ▼'
  }

  return (
    <div>
      <h2 style={{ marginTop: 0 }}>Games</h2>
      <div style={{ marginBottom: 12 }}>
        <strong>Total games:</strong> {total}
      </div>

      <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 12 }}>
        <button onClick={() => setPage((p) => Math.max(0, p - 1))} disabled={page <= 0}>
          Prev
        </button>
        <div>
          Page {page + 1} / {totalPages}
        </div>
        <button onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))} disabled={page >= totalPages - 1}>
          Next
        </button>

        <div style={{ marginLeft: 'auto' }}>
          <label>
            Page size:{' '}
            <select
              value={pageSize}
              onChange={(e) => {
                setPage(0)
                setPageSize(Number(e.target.value))
              }}
            >
              <option value={25}>25</option>
              <option value={50}>50</option>
              <option value={100}>100</option>
              <option value={200}>200</option>
            </select>
          </label>
        </div>
      </div>

      {games.length === 0 ? (
        <div>No games found.</div>
      ) : (
        <table style={{ borderCollapse: 'collapse', width: '100%' }}>
          <thead>
            <tr>
              <th style={{ textAlign: 'left', borderBottom: '1px solid #ddd', padding: 6 }}>
                <button style={{ all: 'unset', cursor: 'pointer' }} onClick={() => toggleSort('game_id')}>
                  Game ID{sortLabel('game_id')}
                </button>
              </th>
              <th style={{ textAlign: 'left', borderBottom: '1px solid #ddd', padding: 6 }}>
                <button style={{ all: 'unset', cursor: 'pointer' }} onClick={() => toggleSort('started_ns')}>
                  Played (local){sortLabel('started_ns')}
                </button>
              </th>
              <th style={{ textAlign: 'right', borderBottom: '1px solid #ddd', padding: 6 }}>
                <button style={{ all: 'unset', cursor: 'pointer' }} onClick={() => toggleSort('turn_count')}>
                  Turns{sortLabel('turn_count')}
                </button>
              </th>
              <th style={{ textAlign: 'right', borderBottom: '1px solid #ddd', padding: 6 }}>
                <button style={{ all: 'unset', cursor: 'pointer' }} onClick={() => toggleSort('width')}>
                  W{sortLabel('width')}
                </button>
              </th>
              <th style={{ textAlign: 'right', borderBottom: '1px solid #ddd', padding: 6 }}>
                <button style={{ all: 'unset', cursor: 'pointer' }} onClick={() => toggleSort('height')}>
                  H{sortLabel('height')}
                </button>
              </th>
              <th style={{ textAlign: 'left', borderBottom: '1px solid #ddd', padding: 6 }}>
                <button style={{ all: 'unset', cursor: 'pointer' }} onClick={() => toggleSort('source')}>
                  Source{sortLabel('source')}
                </button>
              </th>
              <th style={{ textAlign: 'left', borderBottom: '1px solid #ddd', padding: 6 }}>
                <button style={{ all: 'unset', cursor: 'pointer' }} onClick={() => toggleSort('file')}>
                  File{sortLabel('file')}
                </button>
              </th>
            </tr>
          </thead>
          <tbody>
            {games.map((g) => (
              <tr key={g.game_id}>
                <td style={{ padding: 6, borderBottom: '1px solid #f0f0f0' }}>
                  <Link to={`/game/${encodeURIComponent(g.game_id)}`}>{g.game_id}</Link>
                </td>
                <td style={{ padding: 6, borderBottom: '1px solid #f0f0f0' }} title={g.started_ns?.toString() ?? ''}>
                  {formatLocalTimeFromNs(g.started_ns)}
                </td>
                <td style={{ padding: 6, textAlign: 'right', borderBottom: '1px solid #f0f0f0' }}>{g.turn_count}</td>
                <td style={{ padding: 6, textAlign: 'right', borderBottom: '1px solid #f0f0f0' }}>{g.width}</td>
                <td style={{ padding: 6, textAlign: 'right', borderBottom: '1px solid #f0f0f0' }}>{g.height}</td>
                <td style={{ padding: 6, borderBottom: '1px solid #f0f0f0' }}>{g.source}</td>
                <td style={{ padding: 6, borderBottom: '1px solid #f0f0f0' }}>{g.file}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}
