import { useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { fetchStats, type StatsPoint, type StatsResponse } from '../api'

type RangeKey = '1h' | '6h' | '24h' | '7d'
type BucketKey = '1m' | '5m' | '1h' | '1d'

function bucketNs(bucket: BucketKey): number {
  switch (bucket) {
    case '1m':
      return 60 * 1_000_000_000
    case '5m':
      return 5 * 60 * 1_000_000_000
    case '1h':
      return 60 * 60 * 1_000_000_000
    case '1d':
      return 24 * 60 * 60 * 1_000_000_000
  }
}

function rangeMs(range: RangeKey): number {
  switch (range) {
    case '1h':
      return 60 * 60 * 1000
    case '6h':
      return 6 * 60 * 60 * 1000
    case '24h':
      return 24 * 60 * 60 * 1000
    case '7d':
      return 7 * 24 * 60 * 60 * 1000
  }
}

function fmtLocalFromNs(ns: number): string {
  const ms = Math.floor(ns / 1_000_000)
  const d = new Date(ms)
  if (Number.isNaN(d.getTime())) return ''
  return d.toLocaleString()
}

function clamp01(x: number): number {
  if (!Number.isFinite(x)) return 0
  if (x < 0) return 0
  if (x > 1) return 1
  return x
}

function buildSeries(
  fromNs: number,
  toNs: number,
  stepNs: number,
  points: StatsPoint[],
  pick: (p: StatsPoint) => number,
): { x: number; y: number; label: string }[] {
  const byTs = new Map<number, StatsPoint>()
  for (const p of points) byTs.set(p.t_ns, p)

  const out: { x: number; y: number; label: string }[] = []
  for (let t = fromNs; t <= toNs; t += stepNs) {
    const p = byTs.get(t)
    out.push({ x: t, y: p ? pick(p) : 0, label: fmtLocalFromNs(t) })
  }
  return out
}

function MultiLineChart({
  title,
  lines,
  height = 120,
  formatY,
}: {
  title: string
  lines: { name: string; series: { x: number; y: number; label: string }[]; stroke: string; dash?: string }[]
  height?: number
  formatY?: (y: number) => string
}) {
  const width = 900
  const pad = 28
  const svgRef = useRef<SVGSVGElement | null>(null)
  const [hoverIdx, setHoverIdx] = useState<number | null>(null)
  const [hoverClient, setHoverClient] = useState<{ x: number; y: number } | null>(null)

  const all = lines.flatMap((l) => l.series)
  const ys = all.map((p) => p.y)
  const yMin = ys.length ? Math.min(...ys, 0) : 0
  const yMax = ys.length ? Math.max(...ys, 1) : 1

  const baseSeries = lines[0]?.series ?? []
  const xMin = baseSeries.length ? baseSeries[0].x : 0
  const xMax = baseSeries.length ? baseSeries[baseSeries.length - 1].x : 1

  const sx = (x: number) => {
    if (xMax === xMin) return pad
    return pad + ((x - xMin) / (xMax - xMin)) * (width - pad * 2)
  }
  const sy = (y: number) => {
    if (yMax === yMin) return height / 2
    return height - pad - ((y - yMin) / (yMax - yMin)) * (height - pad * 2)
  }

  const yTicks = useMemo(() => {
    const n = 4
    const out: number[] = []
    if (!Number.isFinite(yMin) || !Number.isFinite(yMax)) return out
    if (yMax === yMin) return [yMin]
    for (let i = 0; i <= n; i++) {
      out.push(yMin + (i / n) * (yMax - yMin))
    }
    return out
  }, [yMin, yMax])

  const paths = useMemo(() => {
    return lines.map((line) => {
      const s = line.series
      if (s.length === 0) return { name: line.name, d: '' }
      const d = s
        .map((p, i) => {
          const x = sx(p.x)
          const y = sy(p.y)
          return `${i === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`
        })
        .join(' ')
      return { name: line.name, d }
    })
  }, [lines, xMin, xMax, yMin, yMax])

  const lastValues = lines.map((l) => l.series[l.series.length - 1]?.y).filter((v) => v != null) as number[]
  const lastShown = lastValues.length ? lastValues[0] : null

  const hovered = hoverIdx != null ? baseSeries[hoverIdx] : null
  const hoveredX = hovered ? sx(hovered.x) : null
  const hoveredY = hovered ? sy(hovered.y) : null

  const onMove: React.MouseEventHandler<SVGSVGElement> = (e) => {
    if (baseSeries.length === 0) return
    const svg = svgRef.current
    if (!svg) return
    const rect = svg.getBoundingClientRect()
    const px = e.clientX - rect.left
    const py = e.clientY - rect.top
    setHoverClient({ x: e.clientX, y: e.clientY })

    const vx = (px / rect.width) * width
    const t = (vx - pad) / Math.max(1, width - pad * 2)
    const idx = Math.round(t * (baseSeries.length - 1))
    const clamped = Math.max(0, Math.min(baseSeries.length - 1, idx))
    setHoverIdx(clamped)
  }

  const onLeave: React.MouseEventHandler<SVGSVGElement> = () => {
    setHoverIdx(null)
    setHoverClient(null)
  }

  return (
    <div style={{ marginBottom: 16, position: 'relative' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
        <h3 style={{ margin: '8px 0' }}>{title}</h3>
        {lastShown != null ? <div style={{ color: '#666' }}>{formatY ? formatY(lastShown) : String(lastShown)}</div> : null}
      </div>
      <svg
        ref={svgRef}
        width="100%"
        viewBox={`0 0 ${width} ${height}`}
        style={{ border: '1px solid #ddd', background: '#fff' }}
        onMouseMove={onMove}
        onMouseLeave={onLeave}
      >
        {/* axes */}
        <line x1={pad} y1={pad} x2={pad} y2={height - pad} stroke="#bbb" strokeWidth={1} />
        <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} stroke="#bbb" strokeWidth={1} />

        {/* y ticks */}
        {yTicks.map((v) => {
          const y = sy(v)
          return (
            <g key={v}>
              <line x1={pad - 4} y1={y} x2={pad} y2={y} stroke="#bbb" strokeWidth={1} />
              <text x={pad - 6} y={y + 4} fontSize={10} textAnchor="end" fill="#555">
                {formatY ? formatY(v) : v.toFixed(2)}
              </text>
            </g>
          )
        })}

        {/* x labels (start/end) */}
        {baseSeries.length ? (
          <g>
            <text x={pad} y={height - 8} fontSize={10} textAnchor="start" fill="#777">
              {baseSeries[0].label}
            </text>
            <text x={width - pad} y={height - 8} fontSize={10} textAnchor="end" fill="#777">
              {baseSeries[baseSeries.length - 1].label}
            </text>
          </g>
        ) : null}

        {paths.map((p, i) => (
          <path
            key={p.name}
            d={p.d}
            fill="none"
            stroke={lines[i].stroke}
            strokeWidth={2}
            strokeDasharray={lines[i].dash}
          />
        ))}

        {/* hover crosshair + point */}
        {hovered && hoveredX != null && hoveredY != null ? (
          <g>
            <line x1={hoveredX} y1={pad} x2={hoveredX} y2={height - pad} stroke="#ddd" strokeWidth={1} />
            <circle cx={hoveredX} cy={hoveredY} r={3.5} fill="#111" />
          </g>
        ) : null}
      </svg>

      {hovered && hoverClient ? (
        <div
          style={{
            position: 'fixed',
            left: hoverClient.x + 12,
            top: hoverClient.y + 12,
            background: '#fff',
            border: '1px solid #ddd',
            padding: '6px 8px',
            fontSize: 12,
            color: '#111',
            pointerEvents: 'none',
            whiteSpace: 'nowrap',
          }}
        >
          <div style={{ color: '#555', marginBottom: 2 }}>{hovered.label}</div>
          {lines.map((l) => {
            const v = hoverIdx != null ? l.series[hoverIdx]?.y : null
            if (v == null) return null
            return (
              <div key={l.name}>
                <span style={{ color: '#555' }}>{l.name}:</span> <strong>{formatY ? formatY(v) : String(v)}</strong>
              </div>
            )
          })}
        </div>
      ) : null}
    </div>
  )
}

export default function StatsPage() {
  const [range, setRange] = useState<RangeKey>('24h')
  const [bucket, setBucket] = useState<BucketKey>('5m')
  const [resp, setResp] = useState<StatsResponse | null>(null)
  const [error, setError] = useState<string>('')
  const [loading, setLoading] = useState<boolean>(true)

  useEffect(() => {
    let cancelled = false

    const nowMs = Date.now()
    const fromMs = nowMs - rangeMs(range)
    const fromNs = Math.floor(fromMs * 1_000_000)
    const toNs = Math.floor(nowMs * 1_000_000)
    const bNs = bucketNs(bucket)

    setLoading(true)
    fetchStats(fromNs, toNs, bNs)
      .then((r) => {
        if (cancelled) return
        setResp(r)
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
  }, [range, bucket])

  const points = resp?.points ?? []
  const fromNs = resp?.from_ns ?? 0
  const toNs = resp?.to_ns ?? 0
  const stepNs = resp?.bucket_ns ?? bucketNs(bucket)

  const selfplayGamesSeries = useMemo(
    () => buildSeries(fromNs, toNs, stepNs, points, (p) => p.selfplay_games),
    [fromNs, toNs, stepNs, points],
  )
  const scrapedGamesSeries = useMemo(
    () => buildSeries(fromNs, toNs, stepNs, points, (p) => p.scraped_games),
    [fromNs, toNs, stepNs, points],
  )
  const selfplayTurnsSeries = useMemo(
    () => buildSeries(fromNs, toNs, stepNs, points, (p) => p.selfplay_total_turns),
    [fromNs, toNs, stepNs, points],
  )
  const scrapedTurnsSeries = useMemo(
    () => buildSeries(fromNs, toNs, stepNs, points, (p) => p.scraped_total_turns),
    [fromNs, toNs, stepNs, points],
  )
  const selfplayAvgTurnsSeries = useMemo(
    () =>
      buildSeries(fromNs, toNs, stepNs, points, (p) => {
        if (p.selfplay_games <= 0) return 0
        return p.selfplay_total_turns / p.selfplay_games
      }),
    [fromNs, toNs, stepNs, points],
  )
  const scrapedAvgTurnsSeries = useMemo(
    () =>
      buildSeries(fromNs, toNs, stepNs, points, (p) => {
        if (p.scraped_games <= 0) return 0
        return p.scraped_total_turns / p.scraped_games
      }),
    [fromNs, toNs, stepNs, points],
  )
  const selfplayWinPctSeries = useMemo(
    () =>
      buildSeries(fromNs, toNs, stepNs, points, (p) => {
        const denom = p.selfplay_wins + p.selfplay_draws
        if (denom <= 0) return 0
        return clamp01(p.selfplay_wins / denom)
      }),
    [fromNs, toNs, stepNs, points],
  )
  const scrapedWinPctSeries = useMemo(
    () =>
      buildSeries(fromNs, toNs, stepNs, points, (p) => {
        const denom = p.scraped_wins + p.scraped_draws
        if (denom <= 0) return 0
        return clamp01(p.scraped_wins / denom)
      }),
    [fromNs, toNs, stepNs, points],
  )
  const selfplayDrawPctSeries = useMemo(
    () =>
      buildSeries(fromNs, toNs, stepNs, points, (p) => {
        const denom = p.selfplay_wins + p.selfplay_draws
        if (denom <= 0) return 0
        return clamp01(p.selfplay_draws / denom)
      }),
    [fromNs, toNs, stepNs, points],
  )
  const scrapedDrawPctSeries = useMemo(
    () =>
      buildSeries(fromNs, toNs, stepNs, points, (p) => {
        const denom = p.scraped_wins + p.scraped_draws
        if (denom <= 0) return 0
        return clamp01(p.scraped_draws / denom)
      }),
    [fromNs, toNs, stepNs, points],
  )

  return (
    <div>
      <h2 style={{ marginTop: 0 }}>Stats</h2>
      <div style={{ marginBottom: 8 }}>
        <Link to="/">← Back</Link>
      </div>

      <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 12, flexWrap: 'wrap' }}>
        <label>
          Range:{' '}
          <select value={range} onChange={(e) => setRange(e.target.value as RangeKey)}>
            <option value="1h">Last 1h</option>
            <option value="6h">Last 6h</option>
            <option value="24h">Last 24h</option>
            <option value="7d">Last 7d</option>
          </select>
        </label>
        <label>
          Bucket:{' '}
          <select value={bucket} onChange={(e) => setBucket(e.target.value as BucketKey)}>
            <option value="1m">1m</option>
            <option value="5m">5m</option>
            <option value="1h">1h</option>
            <option value="1d">1d</option>
          </select>
        </label>
        {resp ? (
          <div style={{ color: '#666' }}>
            {fmtLocalFromNs(resp.from_ns)} → {fmtLocalFromNs(resp.to_ns)}
          </div>
        ) : null}
      </div>

      {loading ? <div>Loading…</div> : null}
      {error ? <div style={{ whiteSpace: 'pre-wrap' }}>Error: {error}</div> : null}

      {!loading && !error && resp ? (
        <div>
          <MultiLineChart
            title="Games / bucket"
            lines={[
              { name: 'selfplay', series: selfplayGamesSeries, stroke: '#111' },
              { name: 'scraped', series: scrapedGamesSeries, stroke: '#666', dash: '4 3' },
            ]}
            formatY={(y) => `${Math.round(y)}`}
          />
          <MultiLineChart
            title="Turns / bucket"
            lines={[
              { name: 'selfplay', series: selfplayTurnsSeries, stroke: '#111' },
              { name: 'scraped', series: scrapedTurnsSeries, stroke: '#666', dash: '4 3' },
            ]}
            formatY={(y) => `${Math.round(y)}`}
          />
          <MultiLineChart
            title="Avg turns / game"
            lines={[
              { name: 'selfplay', series: selfplayAvgTurnsSeries, stroke: '#111' },
              { name: 'scraped', series: scrapedAvgTurnsSeries, stroke: '#666', dash: '4 3' },
            ]}
            formatY={(y) => y.toFixed(1)}
          />

          <MultiLineChart
            title="Win %"
            lines={[
              { name: 'selfplay', series: selfplayWinPctSeries, stroke: '#111' },
              { name: 'scraped', series: scrapedWinPctSeries, stroke: '#666', dash: '4 3' },
            ]}
            formatY={(y) => `${Math.round(clamp01(y) * 100)}%`}
          />
          <MultiLineChart
            title="Draw %"
            lines={[
              { name: 'selfplay', series: selfplayDrawPctSeries, stroke: '#111' },
              { name: 'scraped', series: scrapedDrawPctSeries, stroke: '#666', dash: '4 3' },
            ]}
            formatY={(y) => `${Math.round(clamp01(y) * 100)}%`}
          />
        </div>
      ) : null}
    </div>
  )
}
