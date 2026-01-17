import { Link, Route, Routes } from 'react-router-dom'
import GamesPage from './pages/GamesPage'
import GamePage from './pages/GamePage'
import StatsPage from './pages/StatsPage'
import DebugGamePage from './pages/DebugGamePage'

export default function App() {
  return (
    <div style={{ padding: 16, fontFamily: 'system-ui, sans-serif' }}>
      <header style={{ display: 'flex', gap: 12, alignItems: 'baseline' }}>
        <h1 style={{ margin: 0, fontSize: 18 }}>snek2 viewer</h1>
        <Link to="/">Games</Link>
        <Link to="/stats">Stats</Link>
      </header>

      <main style={{ marginTop: 16 }}>
        <Routes>
          <Route path="/" element={<GamesPage />} />
          <Route path="/game/:gameId" element={<GamePage />} />
          <Route path="/debug/:gameId" element={<DebugGamePage />} />
          <Route path="/stats" element={<StatsPage />} />
        </Routes>
      </main>
    </div>
  )
}
