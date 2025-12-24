import { Link, Route, Routes } from 'react-router-dom'
import GamesPage from './pages/GamesPage'
import GamePage from './pages/GamePage'

export default function App() {
  return (
    <div style={{ padding: 16, fontFamily: 'system-ui, sans-serif' }}>
      <header style={{ display: 'flex', gap: 12, alignItems: 'baseline' }}>
        <h1 style={{ margin: 0, fontSize: 18 }}>snek2 viewer</h1>
        <Link to="/">Games</Link>
      </header>

      <main style={{ marginTop: 16 }}>
        <Routes>
          <Route path="/" element={<GamesPage />} />
          <Route path="/game/:gameId" element={<GamePage />} />
        </Routes>
      </main>
    </div>
  )
}
