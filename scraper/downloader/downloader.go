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

func BuildTrainingRows(gameID string, frames []FrameData) []store.TrainingRow {
	outcomes := determineOutcomes(frames)

	var rows []store.TrainingRow

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

		for _, s := range cur.Snakes {
			if s.Death != nil || s.Health <= 0 || len(s.Body) == 0 {
				continue
			}
			nh, ok := nextHead[s.ID]
			if !ok {
				continue
			}
			ch := s.Body[0]
			dx := nh.X - ch.X
			dy := nh.Y - ch.Y

			move := -1
			switch {
			case dy == 1 && dx == 0:
				move = 0 // Up
			case dy == -1 && dx == 0:
				move = 1 // Down
			case dx == -1 && dy == 0:
				move = 2 // Left
			case dx == 1 && dy == 0:
				move = 3 // Right
			}
			if move < 0 {
				continue
			}

			x := EgoStateToBytes(cur, s.ID)
			value := float32(0)
			if v, ok := outcomes[s.ID]; ok {
				value = v
			}

			rows = append(rows, store.TrainingRow{
				GameID: gameID,
				Turn:   int32(cur.Turn),
				EgoID:  s.ID,
				Width:  width,
				Height: height,
				X:      x,
				Policy: int32(move),
				Value:  value,
			})
		}
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
