package main

import (
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/brensch/snek2/scraper/store"
	"github.com/parquet-go/parquet-go"
)

const debugGamesDir = "debug_games"

// listDebugGames returns a list of all available debug games.
func listDebugGames() ([]DebugGameSummary, error) {
	entries, err := os.ReadDir(debugGamesDir)
	if err != nil {
		if os.IsNotExist(err) {
			return []DebugGameSummary{}, nil
		}
		return nil, err
	}

	games := make([]DebugGameSummary, 0)
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		if !strings.HasSuffix(entry.Name(), ".parquet") {
			continue
		}

		// Try to read the first row to get metadata
		filePath := filepath.Join(debugGamesDir, entry.Name())
		summary, err := readDebugGameSummary(filePath)
		if err != nil {
			continue // Skip invalid files
		}
		summary.FileName = entry.Name()
		games = append(games, summary)
	}

	return games, nil
}

// readDebugGameSummary reads just the summary info from a debug game parquet file.
func readDebugGameSummary(filePath string) (DebugGameSummary, error) {
	f, err := os.Open(filePath)
	if err != nil {
		return DebugGameSummary{}, err
	}
	defer f.Close()

	stat, err := f.Stat()
	if err != nil {
		return DebugGameSummary{}, err
	}

	pf, err := parquet.OpenFile(f, stat.Size())
	if err != nil {
		return DebugGameSummary{}, err
	}

	reader := parquet.NewGenericReader[store.DebugTurnRow](pf)
	defer reader.Close()

	// Read first row to get metadata
	rows := make([]store.DebugTurnRow, 1)
	n, err := reader.Read(rows)
	if err != nil && err != io.EOF {
		return DebugGameSummary{}, err
	}
	if n == 0 {
		return DebugGameSummary{}, io.EOF
	}

	return DebugGameSummary{
		GameID:    rows[0].GameID,
		ModelPath: rows[0].ModelPath,
		TurnCount: int(reader.NumRows()),
		Sims:      int(rows[0].Sims),
		Cpuct:     rows[0].Cpuct,
	}, nil
}

// loadDebugGame loads a complete debug game by game ID.
func loadDebugGame(gameID string) (*DebugGameResponse, error) {
	// Find the file
	entries, err := os.ReadDir(debugGamesDir)
	if err != nil {
		return nil, err
	}

	var filePath string
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		if !strings.HasSuffix(entry.Name(), ".parquet") {
			continue
		}
		// Check if this file contains our game
		if strings.Contains(entry.Name(), gameID) {
			filePath = filepath.Join(debugGamesDir, entry.Name())
			break
		}
	}

	if filePath == "" {
		return nil, os.ErrNotExist
	}

	// Read the file
	f, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	stat, err := f.Stat()
	if err != nil {
		return nil, err
	}

	pf, err := parquet.OpenFile(f, stat.Size())
	if err != nil {
		return nil, err
	}

	reader := parquet.NewGenericReader[store.DebugTurnRow](pf)
	defer reader.Close()

	// Read all rows
	allRows := make([]store.DebugTurnRow, 0, int(reader.NumRows()))
	buf := make([]store.DebugTurnRow, 256)
	for {
		n, err := reader.Read(buf)
		if n > 0 {
			allRows = append(allRows, buf[:n]...)
		}
		if err != nil {
			break
		}
	}

	if len(allRows) == 0 {
		return nil, os.ErrNotExist
	}

	// Convert to response format
	resp := &DebugGameResponse{
		GameID:    allRows[0].GameID,
		ModelPath: allRows[0].ModelPath,
		TurnCount: len(allRows),
		Sims:      int(allRows[0].Sims),
		Cpuct:     allRows[0].Cpuct,
		Turns:     make([]DebugTurnData, 0, len(allRows)),
	}

	for _, row := range allRows {
		turnData := DebugTurnData{
			GameID:    row.GameID,
			ModelPath: row.ModelPath,
			Turn:      row.Turn,
			Sims:      int(row.Sims),
			Cpuct:     row.Cpuct,
		}

		// Build state
		snakes := make([]DebugSnakeState, 0, len(row.Snakes))
		for _, s := range row.Snakes {
			body := make([]Point, 0, len(s.BodyX))
			for i := range s.BodyX {
				body = append(body, Point{X: s.BodyX[i], Y: s.BodyY[i]})
			}
			snakes = append(snakes, DebugSnakeState{
				ID:     s.ID,
				Health: s.Health,
				Body:   body,
				Alive:  s.Alive,
			})
		}

		food := make([]Point, 0, len(row.FoodX))
		for i := range row.FoodX {
			food = append(food, Point{X: row.FoodX[i], Y: row.FoodY[i]})
		}

		turnData.State = &DebugGameState{
			Turn:   row.Turn,
			Width:  row.Width,
			Height: row.Height,
			YouId:  "",
			Food:   food,
			Snakes: snakes,
		}

		// Build snake order from alive snakes
		snakeOrder := make([]string, 0, len(snakes))
		for _, s := range snakes {
			if s.Alive {
				snakeOrder = append(snakeOrder, s.ID)
			}
		}
		turnData.SnakeOrder = snakeOrder

		// Parse tree from JSON
		// V2: TreesJSON contains a single DebugMCTSNode (the shared alternating tree)
		// V1 fallback: TreesJSON contains []*DebugSnakeTree
		if len(row.TreesJSON) > 0 {
			// Try V2 format first (single tree object)
			var singleTree *DebugMCTSNode
			if err := json.Unmarshal(row.TreesJSON, &singleTree); err == nil && singleTree != nil {
				turnData.Tree = singleTree
			} else {
				// Fall back to V1 format (array of snake trees)
				var trees []*DebugSnakeTree
				if err := json.Unmarshal(row.TreesJSON, &trees); err == nil {
					turnData.Trees = trees
				}
			}
		}

		resp.Turns = append(resp.Turns, turnData)
	}

	return resp, nil
}
