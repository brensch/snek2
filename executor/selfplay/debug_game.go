package selfplay

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"sort"
	"sync"
	"time"

	"github.com/brensch/snek2/executor/mcts"
	"github.com/brensch/snek2/game"
	"github.com/brensch/snek2/rules"
	"github.com/brensch/snek2/scraper/store"
)

// DebugMCTSNode is a JSON-serializable MCTS node for debug output.
type DebugMCTSNode struct {
	Move       int     `json:"move"`      // 0=Up, 1=Down, 2=Left, 3=Right, -1 for root
	VisitCount int     `json:"n"`         // N
	ValueSum   float32 `json:"value_sum"` // Sum of values
	Q          float32 `json:"q"`         // Average value (ValueSum/N)
	PriorProb  float32 `json:"p"`         // Prior probability
	UCB        float32 `json:"ucb"`       // UCB at last selection

	// Game state at this node
	State *DebugGameState `json:"state,omitempty"`

	// Children (up to 4)
	Children []*DebugMCTSNode `json:"children,omitempty"`
}

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

// DebugSnakeTree holds the MCTS tree for one snake at a turn.
type DebugSnakeTree struct {
	SnakeID  string         `json:"snake_id"`
	SnakeIdx int            `json:"snake_idx"`
	BestMove int            `json:"best_move"`
	Root     *DebugMCTSNode `json:"root"`
}

// DebugTurnData holds all MCTS trees and state for a single turn.
type DebugTurnData struct {
	GameID    string            `json:"game_id"`
	ModelPath string            `json:"model_path"`
	Turn      int32             `json:"turn"`
	Sims      int               `json:"sims"`
	Cpuct     float32           `json:"cpuct"`
	State     *DebugGameState   `json:"state"`
	Trees     []*DebugSnakeTree `json:"trees"`
}

// DebugGameResult holds the result of a debug game generation.
type DebugGameResult struct {
	GameID    string
	ModelPath string
	Turns     []DebugTurnData
	Winner    string
	TurnCount int
}

// DebugProgress is passed to the progress callback after each turn.
type DebugProgress struct {
	Turn       int32
	Moves      map[string]int // snakeID -> move chosen
	AliveCount int
}

// PlayDebugGame generates a game with full MCTS tree capture for debugging.
// The optional onProgress callback is called after each turn completes.
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

		// Capture current state
		turnData := DebugTurnData{
			GameID:    gameID,
			ModelPath: modelPath,
			Turn:      state.Turn,
			Sims:      sims,
			Cpuct:     mctsConfig.Cpuct,
			State:     gameStateToDebug(state),
			Trees:     make([]*DebugSnakeTree, 0, len(state.Snakes)),
		}

		moves := make(map[string]int)
		var movesMu sync.Mutex

		var wg sync.WaitGroup

		// Run MCTS for each snake and capture full tree
		for idx, snake := range state.Snakes {
			if snake.Health <= 0 {
				continue
			}

			wg.Add(1)
			go func(snakeIdx int, snakeID string) {
				defer wg.Done()

				if ctx != nil {
					select {
					case <-ctx.Done():
						return
					default:
					}
				}

				localState := state.Clone()
				localState.YouId = snakeID

				mctsRng := rand.New(rand.NewSource(time.Now().UnixNano()))
				mctsInstance := &mcts.MCTS{
					Config: mctsConfig,
					Client: client,
					Rng:    mctsRng,
				}

				root, _, err := mctsInstance.Search(ctx, localState, sims)
				if err != nil {
					log.Printf("MCTS Error for %s: %v", snakeID, err)
					return
				}

				// Convert MCTS tree to debug format
				debugRoot := convertMCTSTree(root, mctsConfig.Cpuct, -1)

				// Find best move
				bestMove := 0
				bestN := -1
				for i := 0; i < 4; i++ {
					child := root.Children[mcts.Move(i)]
					if child != nil && child.VisitCount > bestN {
						bestN = child.VisitCount
						bestMove = i
					}
				}

				movesMu.Lock()
				moves[snakeID] = bestMove
				turnData.Trees = append(turnData.Trees, &DebugSnakeTree{
					SnakeID:  snakeID,
					SnakeIdx: snakeIdx,
					BestMove: bestMove,
					Root:     debugRoot,
				})
				movesMu.Unlock()
			}(idx, snake.Id)
		}
		wg.Wait()

		// Sort trees by snake index for consistent ordering
		sort.Slice(turnData.Trees, func(i, j int) bool {
			return turnData.Trees[i].SnakeIdx < turnData.Trees[j].SnakeIdx
		})

		result.Turns = append(result.Turns, turnData)

		// Report progress
		if onProgress != nil {
			aliveCount := 0
			for _, s := range state.Snakes {
				if s.Health > 0 {
					aliveCount++
				}
			}
			onProgress(DebugProgress{
				Turn:       state.Turn,
				Moves:      moves,
				AliveCount: aliveCount,
			})
		}

		// Advance state
		state = rules.NextStateSimultaneousWithRNG(state, moves, rng)
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

// convertMCTSTree converts a mcts.Node tree to a DebugMCTSNode tree.
func convertMCTSTree(node *mcts.Node, cpuct float32, move int) *DebugMCTSNode {
	if node == nil {
		return nil
	}

	q := float32(0)
	if node.VisitCount > 0 {
		q = node.ValueSum / float32(node.VisitCount)
	}

	// Calculate UCB for this node (relative to parent's visit count)
	ucb := q + cpuct*node.PriorProb // simplified, parent N not available here

	dn := &DebugMCTSNode{
		Move:       move,
		VisitCount: node.VisitCount,
		ValueSum:   node.ValueSum,
		Q:          q,
		PriorProb:  node.PriorProb,
		UCB:        ucb,
	}

	if node.State != nil {
		dn.State = gameStateToDebug(node.State)
	}

	// Recursively convert children
	for i := 0; i < 4; i++ {
		child := node.Children[mcts.Move(i)]
		if child != nil {
			// Calculate proper UCB for child
			sqrtN := float32(math.Sqrt(float64(node.VisitCount)))
			childQ := float32(0)
			if child.VisitCount > 0 {
				childQ = child.ValueSum / float32(child.VisitCount)
			}
			childUCB := childQ + cpuct*child.PriorProb*sqrtN/(1+float32(child.VisitCount))

			childDebug := convertMCTSTree(child, cpuct, i)
			childDebug.UCB = childUCB
			dn.Children = append(dn.Children, childDebug)
		}
	}

	return dn
}

// gameStateToDebug converts a game.GameState to DebugGameState.
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

// WriteDebugGameParquet writes a debug game result to parquet format.
func WriteDebugGameParquet(outDir string, result *DebugGameResult) (string, error) {
	rows := make([]store.DebugTurnRow, 0, len(result.Turns))

	for _, turn := range result.Turns {
		// Serialize trees to JSON
		treesJSON, err := json.Marshal(turn.Trees)
		if err != nil {
			return "", fmt.Errorf("marshal trees: %w", err)
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

			// Find this snake's policy from trees
			policy := int32(-1)
			for _, tree := range turn.Trees {
				if tree.SnakeID == s.ID {
					policy = int32(tree.BestMove)
					break
				}
			}

			snakes = append(snakes, store.DebugSnake{
				ID:     s.ID,
				Alive:  s.Alive,
				Health: s.Health,
				BodyX:  bodyX,
				BodyY:  bodyY,
				Policy: policy,
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
			TreesJSON: treesJSON,
			Sims:      int32(turn.Sims),
			Cpuct:     turn.Cpuct,
		})
	}

	return store.WriteDebugGameParquet(outDir, result.GameID, rows)
}
