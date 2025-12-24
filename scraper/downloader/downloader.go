package downloader

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"time"

	"github.com/brensch/snek2/scraper/store"
	"github.com/gorilla/websocket"
)

// Config holds downloader configuration
type Config struct {
	EngineURL      string // WebSocket URL template
	ConnectTimeout time.Duration
	ReadTimeout    time.Duration
}

// DefaultConfig returns sensible defaults
func DefaultConfig() Config {
	return Config{
		EngineURL:      "wss://engine.battlesnake.com/games/%s/events",
		ConnectTimeout: 10 * time.Second,
		ReadTimeout:    30 * time.Second,
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
	Turn    int         `json:"turn"`
	Snakes  []SnakeData `json:"snakes"`
	Food    []Coord     `json:"food"`
	Hazards []Coord     `json:"hazards,omitempty"`
	Board   BoardData   `json:"board,omitempty"`
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
	Width   int     `json:"width"`
	Height  int     `json:"height"`
	Hazards []Coord `json:"hazards,omitempty"`
}

type Death struct {
	Cause string `json:"cause"`
	Turn  int    `json:"turn"`
}

// DownloadGame connects to the game WebSocket and downloads all frames.
// It can be cancelled via ctx (e.g. Ctrl+C / SIGTERM).
func DownloadGame(ctx context.Context, gameID string, config Config) ([]FrameData, error) {
	url := fmt.Sprintf(config.EngineURL, gameID)

	dialer := websocket.Dialer{
		HandshakeTimeout: config.ConnectTimeout,
	}

	connectCtx, cancel := context.WithTimeout(ctx, config.ConnectTimeout)
	defer cancel()

	conn, _, err := dialer.DialContext(connectCtx, url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to connect: %w", err)
	}
	defer conn.Close()

	// Ensure ctx cancellation unblocks ReadMessage promptly.
	done := make(chan struct{})
	go func() {
		select {
		case <-ctx.Done():
			_ = conn.WriteControl(websocket.CloseMessage, websocket.FormatCloseMessage(websocket.CloseGoingAway, "shutdown"), time.Now().Add(250*time.Millisecond))
			_ = conn.Close()
		case <-done:
		}
	}()
	defer close(done)

	var frames []FrameData

	// We still parse game_info so we stay compatible with the stream, but we don't persist it.
	var gameInfo GameInfo
	_ = gameInfo

	streamEnded := false

readLoop:
	for {
		select {
		case <-ctx.Done():
			return frames, ctx.Err()
		default:
		}
		if streamEnded {
			break readLoop
		}
		conn.SetReadDeadline(time.Now().Add(config.ReadTimeout))

		_, message, err := conn.ReadMessage()
		if err != nil {
			// Connection closed normally
			if websocket.IsCloseError(err, websocket.CloseNormalClosure, websocket.CloseGoingAway) {
				break
			}
			// Timeout or unexpected close
			if len(frames) > 0 {
				break readLoop
			}
			return nil, fmt.Errorf("read error: %w", err)
		}

		var event GameEvent
		if err := json.Unmarshal(message, &event); err != nil {
			slog.Warn("failed to parse event", "error", err)
			continue
		}

		switch event.Type {
		case "game_info":
			if err := json.Unmarshal(event.Data, &gameInfo); err != nil {
				slog.Warn("failed to parse game_info", "error", err)
			}
		case "frame":
			var frame FrameData
			if err := json.Unmarshal(event.Data, &frame); err != nil {
				slog.Warn("failed to parse frame", "error", err)
				continue
			}
			frames = append(frames, frame)
		case "game_end":
			streamEnded = true
		default:
			// Ignore other event types.
		}
	}

	return frames, nil
}

func BuildArchiveTurns(gameID string, frames []FrameData) []store.ArchiveTurnRow {
	outcomes := determineOutcomes(frames)

	var rows []store.ArchiveTurnRow

	for i := 0; i < len(frames)-1; i++ {
		cur := &frames[i]
		next := &frames[i+1]

		// next heads lookup
		nextHead := make(map[string]Coord, len(next.Snakes))
		for _, s := range next.Snakes {
			if len(s.Body) == 0 {
				continue
			}
			nextHead[s.ID] = s.Body[0]
		}

		width := int32(Width)
		height := int32(Height)
		if cur.Board.Width > 0 {
			width = int32(cur.Board.Width)
		}
		if cur.Board.Height > 0 {
			height = int32(cur.Board.Height)
		}

		turnRow := store.ArchiveTurnRow{
			GameID:  gameID,
			Turn:    int32(cur.Turn),
			Width:   width,
			Height:  height,
			Source:  "scrape",
			FoodX:   nil,
			FoodY:   nil,
			HazardX: nil,
			HazardY: nil,
			Snakes:  nil,
		}

		if len(cur.Food) > 0 {
			turnRow.FoodX = make([]int32, 0, len(cur.Food))
			turnRow.FoodY = make([]int32, 0, len(cur.Food))
			for _, f := range cur.Food {
				turnRow.FoodX = append(turnRow.FoodX, int32(f.X))
				turnRow.FoodY = append(turnRow.FoodY, int32(f.Y))
			}
		}

		// Hazards may appear in both locations depending on feed.
		hzs := 0
		if len(cur.Hazards) > 0 {
			hzs += len(cur.Hazards)
		}
		if len(cur.Board.Hazards) > 0 {
			hzs += len(cur.Board.Hazards)
		}
		if hzs > 0 {
			turnRow.HazardX = make([]int32, 0, hzs)
			turnRow.HazardY = make([]int32, 0, hzs)
			for _, h := range cur.Hazards {
				turnRow.HazardX = append(turnRow.HazardX, int32(h.X))
				turnRow.HazardY = append(turnRow.HazardY, int32(h.Y))
			}
			for _, h := range cur.Board.Hazards {
				turnRow.HazardX = append(turnRow.HazardX, int32(h.X))
				turnRow.HazardY = append(turnRow.HazardY, int32(h.Y))
			}
		}

		turnRow.Snakes = make([]store.ArchiveSnake, 0, len(cur.Snakes))
		for _, s := range cur.Snakes {
			alive := s.Death == nil && s.Health > 0 && len(s.Body) > 0

			policy := int32(-1)
			if alive {
				nh, ok := nextHead[s.ID]
				if ok {
					ch := s.Body[0]
					dx := nh.X - ch.X
					dy := nh.Y - ch.Y
					switch {
					case dy == 1 && dx == 0:
						policy = 0 // Up
					case dy == -1 && dx == 0:
						policy = 1 // Down
					case dx == -1 && dy == 0:
						policy = 2 // Left
					case dx == 1 && dy == 0:
						policy = 3 // Right
					}
				}
			}

			value := float32(0)
			if v, ok := outcomes[s.ID]; ok {
				value = v
			}

			snakeRow := store.ArchiveSnake{
				ID:     s.ID,
				Alive:  alive,
				Health: int32(s.Health),
				Policy: policy,
				Value:  value,
			}
			if len(s.Body) > 0 {
				snakeRow.BodyX = make([]int32, 0, len(s.Body))
				snakeRow.BodyY = make([]int32, 0, len(s.Body))
				for _, bp := range s.Body {
					snakeRow.BodyX = append(snakeRow.BodyX, int32(bp.X))
					snakeRow.BodyY = append(snakeRow.BodyY, int32(bp.Y))
				}
			}
			turnRow.Snakes = append(turnRow.Snakes, snakeRow)
		}

		rows = append(rows, turnRow)
	}

	return rows
}

func determineOutcomes(frames []FrameData) map[string]float32 {
	outcomes := make(map[string]float32)
	if len(frames) == 0 {
		return outcomes
	}

	last := frames[len(frames)-1]

	all := make(map[string]bool)
	for _, f := range frames {
		for _, s := range f.Snakes {
			all[s.ID] = true
		}
	}

	alive := make(map[string]bool)
	aliveCount := 0
	for _, s := range last.Snakes {
		if s.Death == nil && s.Health > 0 {
			alive[s.ID] = true
			aliveCount++
		}
	}

	// Value target in [-1..1]
	// - solo winner: 1
	// - draw / multiple alive at end: 0
	// - dead: -1
	for id := range all {
		if aliveCount == 1 && alive[id] {
			outcomes[id] = 1.0
			continue
		}
		if aliveCount == 0 {
			outcomes[id] = 0.0
			continue
		}
		if alive[id] {
			outcomes[id] = 0.0
			continue
		}
		outcomes[id] = -1.0
	}

	return outcomes
}
