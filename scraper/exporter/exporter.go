package exporter

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"sort"

	pb "github.com/brensch/snek2/gen/go"
	"github.com/brensch/snek2/go-worker/convert"
	"github.com/brensch/snek2/scraper/db"
	"github.com/brensch/snek2/scraper/downloader"
	"google.golang.org/protobuf/proto"
)

// Exporter converts scraped games to the training format
type Exporter struct {
	db *db.DB
}

// NewExporter creates a new exporter
func NewExporter(database *db.DB) *Exporter {
	return &Exporter{db: database}
}

// ConvertGame converts a game's frames to TrainingExamples
func (e *Exporter) ConvertGame(gameID string) ([]*pb.TrainingExample, error) {
	frames, err := e.db.GetGameFrames(gameID)
	if err != nil {
		return nil, fmt.Errorf("failed to get frames: %w", err)
	}

	if len(frames) == 0 {
		return nil, fmt.Errorf("no frames found for game %s", gameID)
	}

	var examples []*pb.TrainingExample

	// Parse all frames to determine final outcomes
	var parsedFrames []downloader.FrameData
	for _, frame := range frames {
		var data downloader.FrameData
		if err := json.Unmarshal([]byte(frame.RawJSON), &data); err != nil {
			log.Printf("Failed to parse frame %d: %v", frame.Turn, err)
			continue
		}
		parsedFrames = append(parsedFrames, data)
	}

	if len(parsedFrames) == 0 {
		return nil, fmt.Errorf("no valid frames parsed")
	}

	// Determine final outcomes (who won/lost)
	outcomes := determineOutcomes(parsedFrames)

	// Convert each frame to a training example
	for i, frameData := range parsedFrames {
		if i >= len(parsedFrames)-1 {
			// Skip final frame (no moves to learn from)
			continue
		}

		// Create GameState from frame
		state, snakeOrder := frameToGameState(&frameData)
		if state == nil || len(state.Snakes) == 0 {
			continue
		}

		// Get policies from the next frame (what moves were actually made)
		nextFrame := &parsedFrames[i+1]
		policies := extractPolicies(&frameData, nextFrame, snakeOrder)

		// Get values from outcomes
		values := make([]float32, 4)
		for j, snakeID := range snakeOrder {
			if outcome, ok := outcomes[snakeID]; ok {
				values[j] = outcome
			}
		}

		// Convert state to bytes
		stateBytes := convert.StateToBytes(state)
		if stateBytes == nil {
			continue
		}

		example := &pb.TrainingExample{
			StateData: *stateBytes,
			Policies:  policies,
			Values:    values,
		}
		examples = append(examples, example)

		convert.PutBuffer(stateBytes)
	}

	return examples, nil
}

// frameToGameState converts a scraped frame to a GameState
func frameToGameState(frame *downloader.FrameData) (*pb.GameState, []string) {
	state := &pb.GameState{
		Width:  11, // Standard board size
		Height: 11,
		Turn:   int32(frame.Turn),
	}

	// Override if board info is available
	if frame.Board.Width > 0 {
		state.Width = int32(frame.Board.Width)
	}
	if frame.Board.Height > 0 {
		state.Height = int32(frame.Board.Height)
	}

	// Add food
	for _, f := range frame.Food {
		state.Food = append(state.Food, &pb.Point{
			X: int32(f.X),
			Y: int32(f.Y),
		})
	}

	// Sort snakes by ID for consistent ordering
	sortedSnakes := make([]downloader.SnakeData, len(frame.Snakes))
	copy(sortedSnakes, frame.Snakes)
	sort.Slice(sortedSnakes, func(i, j int) bool {
		return sortedSnakes[i].ID < sortedSnakes[j].ID
	})

	var snakeOrder []string
	for _, s := range sortedSnakes {
		if s.Death != nil {
			// Skip dead snakes
			continue
		}

		snake := &pb.Snake{
			Id:     s.ID,
			Health: int32(s.Health),
		}

		for _, p := range s.Body {
			snake.Body = append(snake.Body, &pb.Point{
				X: int32(p.X),
				Y: int32(p.Y),
			})
		}

		state.Snakes = append(state.Snakes, snake)
		snakeOrder = append(snakeOrder, s.ID)
	}

	// Pad snake order to 4
	for len(snakeOrder) < 4 {
		snakeOrder = append(snakeOrder, "")
	}

	return state, snakeOrder
}

// extractPolicies determines what moves were made by comparing consecutive frames
func extractPolicies(current, next *downloader.FrameData, snakeOrder []string) []float32 {
	policies := make([]float32, 16) // 4 snakes * 4 moves

	// Build lookup of current head positions
	currentHeads := make(map[string]downloader.Coord)
	for _, s := range current.Snakes {
		if len(s.Body) > 0 && s.Death == nil {
			currentHeads[s.ID] = s.Body[0]
		}
	}

	// Build lookup of next head positions
	nextHeads := make(map[string]downloader.Coord)
	for _, s := range next.Snakes {
		if len(s.Body) > 0 {
			nextHeads[s.ID] = s.Body[0]
		}
	}

	// Determine moves for each snake
	for i, snakeID := range snakeOrder {
		if snakeID == "" {
			continue
		}

		currentHead, ok1 := currentHeads[snakeID]
		nextHead, ok2 := nextHeads[snakeID]

		if !ok1 || !ok2 {
			// Snake died or wasn't present
			continue
		}

		// Calculate move direction
		dx := nextHead.X - currentHead.X
		dy := nextHead.Y - currentHead.Y

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

		if move >= 0 {
			policies[i*4+move] = 1.0
		}
	}

	return policies
}

// determineOutcomes analyzes the game to determine win/loss values for each snake
func determineOutcomes(frames []downloader.FrameData) map[string]float32 {
	outcomes := make(map[string]float32)

	if len(frames) == 0 {
		return outcomes
	}

	lastFrame := frames[len(frames)-1]

	// Find all snakes that appeared in the game
	allSnakes := make(map[string]bool)
	for _, frame := range frames {
		for _, s := range frame.Snakes {
			allSnakes[s.ID] = true
		}
	}

	// Find alive snakes in final frame
	aliveCount := 0
	var aliveSnakes []string
	for _, s := range lastFrame.Snakes {
		if s.Death == nil && s.Health > 0 {
			aliveCount++
			aliveSnakes = append(aliveSnakes, s.ID)
		}
	}

	// Assign outcomes
	for snakeID := range allSnakes {
		isAlive := false
		for _, alive := range aliveSnakes {
			if alive == snakeID {
				isAlive = true
				break
			}
		}

		if aliveCount == 1 && isAlive {
			// Solo winner
			outcomes[snakeID] = 1.0
		} else if aliveCount == 0 {
			// Draw - everyone died
			outcomes[snakeID] = 0.5
		} else if isAlive {
			// Game ended with multiple alive (max turns?)
			outcomes[snakeID] = 0.5
		} else {
			// Dead snake
			outcomes[snakeID] = 0.0
		}
	}

	return outcomes
}

// ExportToProto converts and exports unprocessed games to a TrainingData proto
func (e *Exporter) ExportToProto(maxGames int) (*pb.TrainingData, error) {
	games, err := e.db.GetUnprocessedGames(maxGames)
	if err != nil {
		return nil, fmt.Errorf("failed to get unprocessed games: %w", err)
	}

	trainingData := &pb.TrainingData{}

	for _, game := range games {
		examples, err := e.ConvertGame(game.ID)
		if err != nil {
			log.Printf("Failed to convert game %s: %v", game.ID, err)
			continue
		}

		trainingData.Examples = append(trainingData.Examples, examples...)

		// Mark as processed
		if err := e.db.MarkGameProcessed(game.ID); err != nil {
			log.Printf("Failed to mark game %s as processed: %v", game.ID, err)
		}
	}

	return trainingData, nil
}

// ExportToFile exports training data to a protobuf file
func (e *Exporter) ExportToFile(maxGames int, outputPath string) error {
	trainingData, err := e.ExportToProto(maxGames)
	if err != nil {
		return err
	}

	data, err := proto.Marshal(trainingData)
	if err != nil {
		return fmt.Errorf("failed to marshal training data: %w", err)
	}

	if err := os.WriteFile(outputPath, data, 0644); err != nil {
		return fmt.Errorf("failed to write file: %w", err)
	}

	log.Printf("Exported %d examples to %s", len(trainingData.Examples), outputPath)
	return nil
}
