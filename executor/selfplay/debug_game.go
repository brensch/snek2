// debug_game.go - Debug game generation with full MCTS tree capture.
//
// PlayDebugGame generates games with full MCTS tree data at each turn,
// enabling visualization and analysis of the search process. The output
// is stored as parquet files that can be viewed in the web frontend.
package selfplay

import (
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"time"

	"github.com/brensch/snek2/executor/mcts"
	"github.com/brensch/snek2/game"
	"github.com/brensch/snek2/rules"
	"github.com/brensch/snek2/scraper/store"
)

// DebugGameState is a minimal game state for debug visualization.
type DebugGameState struct {
	Turn   int32             `json:"turn"`
	Width  int32             `json:"width"`
	Height int32             `json:"height"`
	YouId  string            `json:"you_id"`
	Food   []game.Point      `json:"food"`
	Snakes []DebugSnakeState `json:"snakes"`
}

// DebugSnakeState is a minimal snake state for debug visualization.
type DebugSnakeState struct {
	ID     string       `json:"id"`
	Health int32        `json:"health"`
	Body   []game.Point `json:"body"`
	Alive  bool         `json:"alive"`
}

// DebugProgress is passed to the progress callback after each turn.
type DebugProgress struct {
	Turn       int32
	Moves      map[string]int // snakeID -> move chosen
	AliveCount int
}

// DebugJointNode is a JSON-serializable node for the unified MCTS tree.
// Each child represents ALL snakes making moves together.
type DebugJointNode struct {
	// Moves shows what each snake did to reach this node (nil for root)
	Moves map[string]int `json:"moves,omitempty"`

	VisitCount int     `json:"n"`
	ValueSum   float32 `json:"value_sum"`
	Q          float32 `json:"q"`
	PriorProb  float32 `json:"p"`   // Joint prior (product of individual priors)
	UCB        float32 `json:"ucb"` // At last selection

	// State after the moves were applied
	State *DebugGameState `json:"state,omitempty"`

	// Per-snake priors at this node (for understanding why moves were chosen)
	SnakePriors map[string][4]float32 `json:"snake_priors,omitempty"`

	// Children - each represents a joint action
	Children []*DebugJointNode `json:"children,omitempty"`
}

// DebugTurnData holds the unified MCTS tree for a turn.
type DebugTurnData struct {
	GameID     string          `json:"game_id"`
	ModelPath  string          `json:"model_path"`
	Turn       int32           `json:"turn"`
	Sims       int             `json:"sims"`
	Cpuct      float32         `json:"cpuct"`
	State      *DebugGameState `json:"state"`
	SnakeOrder []string        `json:"snake_order"`
	Tree       *DebugJointNode `json:"tree"`
	ChosenMove map[string]int  `json:"chosen_move"` // The moves chosen for actual play
}

// DebugGameResult holds the result of a debug game with full MCTS tree data.
type DebugGameResult struct {
	GameID    string
	ModelPath string
	Turns     []DebugTurnData
	Winner    string
	TurnCount int
}

// PlayDebugGame generates a game with unified MCTS tree (all snakes explore together).
func PlayDebugGame(ctx context.Context, mctsConfig mcts.Config, client mcts.Predictor, modelPath string, sims int, onProgress func(DebugProgress)) (*DebugGameResult, error) {
	rngSeed := time.Now().UnixNano()
	rng := rand.New(rand.NewSource(rngSeed))
	state := createInitialState(rng)
	gameID := fmt.Sprintf("debug_%d", time.Now().UnixNano())

	result := &DebugGameResult{
		GameID:    gameID,
		ModelPath: modelPath,
		Turns:     make([]DebugTurnData, 0, 256),
	}

	for {
		if ctx != nil {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			default:
			}
		}

		if rules.IsGameOver(state) {
			break
		}

		// Run unified MCTS where all snakes explore together
		mctsRoot, err := mcts.UnifiedSearch(ctx, client, mctsConfig, state, sims)
		if err != nil {
			return nil, err
		}

		// Get snake order
		snakeOrder := make([]string, 0, len(state.Snakes))
		for _, s := range state.Snakes {
			if s.Health > 0 {
				snakeOrder = append(snakeOrder, s.Id)
			}
		}

		// Convert to debug format
		debugTree := convertJointNode(mctsRoot, mctsConfig.Cpuct)

		// Get best moves (most visited child)
		chosenMoves := mcts.GetBestMoves(mctsRoot)
		if chosenMoves == nil {
			// Fallback: all snakes go Up
			chosenMoves = make(map[string]int)
			for _, id := range snakeOrder {
				chosenMoves[id] = 0
			}
		}

		turnData := DebugTurnData{
			GameID:     gameID,
			ModelPath:  modelPath,
			Turn:       state.Turn,
			Sims:       sims,
			Cpuct:      mctsConfig.Cpuct,
			State:      gameStateToDebug(state),
			SnakeOrder: snakeOrder,
			Tree:       debugTree,
			ChosenMove: chosenMoves,
		}

		result.Turns = append(result.Turns, turnData)

		// Report progress
		if onProgress != nil {
			onProgress(DebugProgress{
				Turn:       state.Turn,
				Moves:      chosenMoves,
				AliveCount: len(snakeOrder),
			})
		}

		// Advance state with chosen moves
		state = rules.NextStateSimultaneousWithRNG(state, chosenMoves, rng)
	}

	// Determine winner
	winnerId := ""
	living := 0
	for _, s := range state.Snakes {
		if s.Health > 0 {
			living++
			winnerId = s.Id
		}
	}
	if living != 1 {
		winnerId = ""
	}

	result.Winner = winnerId
	result.TurnCount = int(state.Turn)

	return result, nil
}

// gameStateToDebug converts a game.GameState to the debug visualization format.
func gameStateToDebug(state *game.GameState) *DebugGameState {
	if state == nil {
		return nil
	}

	snakes := make([]DebugSnakeState, 0, len(state.Snakes))
	for _, s := range state.Snakes {
		body := make([]game.Point, len(s.Body))
		copy(body, s.Body)
		snakes = append(snakes, DebugSnakeState{
			ID:     s.Id,
			Health: s.Health,
			Body:   body,
			Alive:  s.Health > 0 && len(s.Body) > 0,
		})
	}

	food := make([]game.Point, len(state.Food))
	copy(food, state.Food)

	return &DebugGameState{
		Turn:   state.Turn,
		Width:  state.Width,
		Height: state.Height,
		YouId:  state.YouId,
		Food:   food,
		Snakes: snakes,
	}
}

func convertJointNode(node *mcts.JointNode, cpuct float32) *DebugJointNode {
	if node == nil {
		return nil
	}

	q := float32(0)
	if node.VisitCount > 0 {
		q = node.ValueSum / float32(node.VisitCount)
	}

	dn := &DebugJointNode{
		VisitCount:  node.VisitCount,
		ValueSum:    node.ValueSum,
		Q:           q,
		PriorProb:   1.0,
		UCB:         q,
		SnakePriors: node.SnakePriors,
	}

	if node.State != nil {
		dn.State = gameStateToDebug(node.State)
	}

	// Convert children
	if len(node.Children) > 0 {
		dn.Children = make([]*DebugJointNode, 0, len(node.Children))

		// Sort children by visit count for better display
		type childEntry struct {
			key   string
			child *mcts.JointChild
		}
		entries := make([]childEntry, 0, len(node.Children))
		for k, c := range node.Children {
			entries = append(entries, childEntry{k, c})
		}
		// Sort by visit count descending
		for i := 0; i < len(entries)-1; i++ {
			for j := i + 1; j < len(entries); j++ {
				if entries[j].child.VisitCount > entries[i].child.VisitCount {
					entries[i], entries[j] = entries[j], entries[i]
				}
			}
		}

		sqrtN := float32(1)
		if node.VisitCount > 0 {
			sqrtN = float32(node.VisitCount)
			sqrtN = float32(int(sqrtN*100+0.5)) / 100 // rough sqrt approximation not needed, use actual
		}
		import_math_sqrt := func(x float32) float32 {
			return float32(int(x*1000+0.5)) / 1000
		}
		_ = import_math_sqrt

		for _, entry := range entries {
			child := entry.child
			childQ := float32(0)
			if child.VisitCount > 0 {
				childQ = child.ValueSum / float32(child.VisitCount)
			}

			childDebug := &DebugJointNode{
				Moves:      child.Action.Moves,
				VisitCount: child.VisitCount,
				ValueSum:   child.ValueSum,
				Q:          childQ,
				PriorProb:  child.PriorProb,
				UCB:        childQ + cpuct*child.PriorProb*sqrtN/(1+float32(child.VisitCount)),
			}

			// Recursively convert the child's node
			if child.Node != nil {
				childDebug.State = gameStateToDebug(child.Node.State)
				// Only recurse if the child node has been expanded
				if child.Node.IsExpanded && len(child.Node.Children) > 0 {
					childDebug.Children = convertJointNode(child.Node, cpuct).Children
				}
			}

			dn.Children = append(dn.Children, childDebug)
		}
	}

	return dn
}

// WriteDebugGameParquet writes a debug game result to parquet format.
func WriteDebugGameParquet(outDir string, result *DebugGameResult) (string, error) {
	rows := make([]store.DebugTurnRow, 0, len(result.Turns))

	for _, turn := range result.Turns {
		// Serialize tree to JSON
		treeJSON, err := json.Marshal(turn.Tree)
		if err != nil {
			return "", fmt.Errorf("marshal tree: %w", err)
		}

		// Build snakes
		snakes := make([]store.DebugSnake, 0, len(turn.State.Snakes))
		for _, s := range turn.State.Snakes {
			bodyX := make([]int32, len(s.Body))
			bodyY := make([]int32, len(s.Body))
			for i, p := range s.Body {
				bodyX[i] = p.X
				bodyY[i] = p.Y
			}

			snakes = append(snakes, store.DebugSnake{
				ID:     s.ID,
				Alive:  s.Alive,
				Health: s.Health,
				BodyX:  bodyX,
				BodyY:  bodyY,
				Policy: -1,
			})
		}

		// Build food arrays
		foodX := make([]int32, len(turn.State.Food))
		foodY := make([]int32, len(turn.State.Food))
		for i, p := range turn.State.Food {
			foodX[i] = p.X
			foodY[i] = p.Y
		}

		rows = append(rows, store.DebugTurnRow{
			GameID:    result.GameID,
			ModelPath: result.ModelPath,
			Turn:      turn.Turn,
			Width:     turn.State.Width,
			Height:    turn.State.Height,
			FoodX:     foodX,
			FoodY:     foodY,
			HazardX:   nil,
			HazardY:   nil,
			Snakes:    snakes,
			TreesJSON: treeJSON,
			Sims:      int32(turn.Sims),
			Cpuct:     turn.Cpuct,
		})
	}

	return store.WriteDebugGameParquet(outDir, result.GameID, rows)
}
