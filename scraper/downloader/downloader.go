package downloader

import (
	"encoding/json"
	"fmt"
	"log"
	"sync"
	"sync/atomic"
	"time"

	"github.com/brensch/snek2/scraper/db"
	"github.com/gorilla/websocket"
)

// Config holds downloader configuration
type Config struct {
	NumWorkers     int
	EngineURL      string // WebSocket URL template
	ConnectTimeout time.Duration
	ReadTimeout    time.Duration
}

// DefaultConfig returns sensible defaults
func DefaultConfig() Config {
	return Config{
		NumWorkers:     4,
		EngineURL:      "wss://engine.battlesnake.com/games/%s/events",
		ConnectTimeout: 10 * time.Second,
		ReadTimeout:    30 * time.Second,
	}
}

// Stats holds download statistics
type Stats struct {
	GamesDownloaded int64
	GamesSkipped    int64
	GamesFailed     int64
	FramesTotal     int64
}

// Worker manages a pool of game downloaders
type Worker struct {
	config Config
	db     *db.DB
	stats  Stats
}

// NewWorker creates a new download worker pool
func NewWorker(config Config, database *db.DB) *Worker {
	return &Worker{
		config: config,
		db:     database,
	}
}

// Start begins processing game IDs from the channel
func (w *Worker) Start(gameIDChan <-chan string, done chan<- struct{}) {
	var wg sync.WaitGroup

	// Start worker pool
	for i := 0; i < w.config.NumWorkers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			w.worker(workerID, gameIDChan)
		}(i)
	}

	// Wait for all workers to finish
	wg.Wait()
	close(done)
}

// worker processes game IDs from the channel
func (w *Worker) worker(id int, gameIDChan <-chan string) {
	for gameID := range gameIDChan {
		// Check if game already exists
		exists, err := w.db.GameExists(gameID)
		if err != nil {
			log.Printf("[Worker %d] Error checking game %s: %v", id, gameID, err)
			continue
		}
		if exists {
			atomic.AddInt64(&w.stats.GamesSkipped, 1)
			continue
		}

		// Download the game
		game, frames, err := w.downloadGame(gameID)
		if err != nil {
			log.Printf("[Worker %d] Failed to download %s: %v", id, gameID, err)
			atomic.AddInt64(&w.stats.GamesFailed, 1)
			continue
		}

		// Store in database
		if err := w.db.InsertGame(game, frames); err != nil {
			log.Printf("[Worker %d] Failed to store %s: %v", id, gameID, err)
			atomic.AddInt64(&w.stats.GamesFailed, 1)
			continue
		}

		atomic.AddInt64(&w.stats.GamesDownloaded, 1)
		atomic.AddInt64(&w.stats.FramesTotal, int64(len(frames)))
		log.Printf("[Worker %d] Downloaded %s: %d turns, winner: %s", id, gameID, len(frames), game.Winner)
	}
}

// GameEvent represents an event from the WebSocket stream
type GameEvent struct {
	Type string          `json:"type"`
	Data json.RawMessage `json:"data"`
}

// GameInfo from the "game_info" event
type GameInfo struct {
	Game    GameDetails `json:"game"`
	Ruleset RulesetInfo `json:"ruleset"`
}

type GameDetails struct {
	ID      string `json:"id"`
	Timeout int    `json:"timeout"`
}

type RulesetInfo struct {
	Name     string          `json:"name"`
	Version  string          `json:"version"`
	Settings json.RawMessage `json:"settings"`
}

// FrameData from "frame" events
type FrameData struct {
	Turn   int         `json:"turn"`
	Snakes []SnakeData `json:"snakes"`
	Food   []Coord     `json:"food"`
	Board  BoardData   `json:"board,omitempty"`
}

type SnakeData struct {
	ID     string  `json:"id"`
	Name   string  `json:"name"`
	Health int     `json:"health"`
	Body   []Coord `json:"body"`
	Author string  `json:"author,omitempty"`
	Death  *Death  `json:"death,omitempty"`
}

type Coord struct {
	X int `json:"x"`
	Y int `json:"y"`
}

type BoardData struct {
	Width  int `json:"width"`
	Height int `json:"height"`
}

type Death struct {
	Cause string `json:"cause"`
	Turn  int    `json:"turn"`
}

// downloadGame connects to the game WebSocket and downloads all frames
func (w *Worker) downloadGame(gameID string) (db.Game, []db.Frame, error) {
	url := fmt.Sprintf(w.config.EngineURL, gameID)

	dialer := websocket.Dialer{
		HandshakeTimeout: w.config.ConnectTimeout,
	}

	conn, _, err := dialer.Dial(url, nil)
	if err != nil {
		return db.Game{}, nil, fmt.Errorf("failed to connect: %w", err)
	}
	defer conn.Close()

	var frames []db.Frame
	var gameInfo GameInfo
	var lastFrame *FrameData

	for {
		conn.SetReadDeadline(time.Now().Add(w.config.ReadTimeout))

		_, message, err := conn.ReadMessage()
		if err != nil {
			// Connection closed normally
			if websocket.IsCloseError(err, websocket.CloseNormalClosure, websocket.CloseGoingAway) {
				break
			}
			// Timeout or unexpected close
			if len(frames) > 0 {
				// We got some data, use it
				break
			}
			return db.Game{}, nil, fmt.Errorf("read error: %w", err)
		}

		var event GameEvent
		if err := json.Unmarshal(message, &event); err != nil {
			log.Printf("Failed to parse event: %v", err)
			continue
		}

		switch event.Type {
		case "game_info":
			if err := json.Unmarshal(event.Data, &gameInfo); err != nil {
				log.Printf("Failed to parse game_info: %v", err)
			}

		case "frame":
			var frameData FrameData
			if err := json.Unmarshal(event.Data, &frameData); err != nil {
				log.Printf("Failed to parse frame: %v", err)
				continue
			}

			frames = append(frames, db.Frame{
				GameID:  gameID,
				Turn:    frameData.Turn,
				RawJSON: string(event.Data),
			})
			lastFrame = &frameData

		case "game_end":
			// Game is complete
			break
		}
	}

	// Determine winner from last frame
	winner := determineWinner(lastFrame)

	game := db.Game{
		ID:      gameID,
		Winner:  winner,
		Ruleset: gameInfo.Ruleset.Name,
	}

	return game, frames, nil
}

// determineWinner analyzes the final frame to find the winner
func determineWinner(frame *FrameData) string {
	if frame == nil {
		return "unknown"
	}

	var alive []SnakeData
	for _, snake := range frame.Snakes {
		if snake.Death == nil && snake.Health > 0 {
			alive = append(alive, snake)
		}
	}

	if len(alive) == 1 {
		return alive[0].Name
	} else if len(alive) == 0 {
		return "draw"
	}

	// Multiple alive at end - might be max turns or simultaneous death
	return "draw"
}

// GetStats returns current statistics
func (w *Worker) GetStats() Stats {
	return Stats{
		GamesDownloaded: atomic.LoadInt64(&w.stats.GamesDownloaded),
		GamesSkipped:    atomic.LoadInt64(&w.stats.GamesSkipped),
		GamesFailed:     atomic.LoadInt64(&w.stats.GamesFailed),
		FramesTotal:     atomic.LoadInt64(&w.stats.FramesTotal),
	}
}
