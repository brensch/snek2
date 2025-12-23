package selfplay

import (
	"context"
	"fmt"
	"log"
	"math/rand"
	"sort"
	"sync"
	"time"

	"github.com/brensch/snek2/executor/convert"
	"github.com/brensch/snek2/executor/mcts"
	"github.com/brensch/snek2/game"
	"github.com/brensch/snek2/rules"
	"github.com/brensch/snek2/scraper/store"
)

type GameResult struct {
	WinnerId string
	Steps    int
}

func PlayGame(ctx context.Context, workerId int, mctsConfig mcts.Config, client mcts.Predictor, verbose bool, onStep func()) ([]store.TrainingRow, GameResult) {
	rng := rand.New(rand.NewSource(time.Now().UnixNano() + int64(workerId)*1000003))
	state := createInitialState(rng)
	gameID := fmt.Sprintf("selfplay_%d_%d", time.Now().UnixNano(), workerId)

	rows := make([]store.TrainingRow, 0, 1024)

	moveNames := []string{"Up", "Down", "Left", "Right"}

	for {
		if ctx != nil {
			select {
			case <-ctx.Done():
				return nil, GameResult{WinnerId: "", Steps: int(state.Turn)}
			default:
			}
		}

		if verbose {
			PrintBoard(state)
		}

		if rules.IsGameOver(state) {
			break
		}

		moves := make(map[string]int)
		var movesMu sync.Mutex

		policies := make(map[string][]float32)
		var policiesMu sync.Mutex

		var wg sync.WaitGroup

		// Iterate over all snakes to get their moves
		for _, snake := range state.Snakes {
			if snake.Health <= 0 {
				continue
			}

			snakeID := snake.Id

			wg.Add(1)
			go func(id string) {
				defer wg.Done()

				if ctx != nil {
					select {
					case <-ctx.Done():
						return
					default:
					}
				}

				// Deep copy state for this snake's perspective.
				localState := state.Clone()
				localState.YouId = id

				// MCTS Search
				mctsInstance := &mcts.MCTS{
					Config: mctsConfig,
					Client: client,
				}
				root, _, err := mctsInstance.Search(ctx, localState, 800)
				if err != nil {
					if ctx != nil {
						select {
						case <-ctx.Done():
							return
						default:
						}
					}
					log.Printf("MCTS Error: %v", err)
					return
				}

				// Extract Policy
				policy := make([]float32, 4)
				totalVisits := 0
				for _, child := range root.Children {
					if child != nil {
						totalVisits += child.VisitCount
					}
				}

				// If no valid moves found (e.g. trapped), pick a default
				if totalVisits == 0 {
					movesMu.Lock()
					moves[id] = 0 // Default Up
					movesMu.Unlock()
					return
				}

				for move, child := range root.Children {
					if child != nil && int(move) < len(policy) {
						policy[int(move)] = float32(child.VisitCount) / float32(totalVisits)
					}
				}

				// Select Move
				var move int
				// Use a local RNG or lock the shared one?
				// For simplicity, let's just use the shared one with a lock, or create a new one.
				// Creating a new one is safer for concurrency.
				localRng := rand.New(rand.NewSource(time.Now().UnixNano()))

				if localState.Turn < 10 {
					move = sampleMove(localRng, policy)
				} else {
					move = argmax(policy)
				}

				movesMu.Lock()
				moves[id] = move
				movesMu.Unlock()

				policiesMu.Lock()
				policies[id] = policy
				policiesMu.Unlock()

				if verbose {
					// Per-snake verbose logging is handled after all moves are chosen.
				}
			}(snakeID)
		}
		wg.Wait()

		if ctx != nil {
			select {
			case <-ctx.Done():
				return nil, GameResult{WinnerId: "", Steps: int(state.Turn)}
			default:
			}
		}

		if verbose {
			PrintBoard(state)
			ids := make([]string, 0, len(moves))
			for id := range moves {
				ids = append(ids, id)
			}
			sort.Strings(ids)
			for _, id := range ids {
				m := moves[id]
				moveName := "Unknown"
				if m >= 0 && m < len(moveNames) {
					moveName = moveNames[m]
				}
				log.Printf("[Worker %d] Turn %d: %s -> %s", workerId, state.Turn, id, moveName)
			}
		}

		// Record per-snake supervised samples for this turn (ego-centric).
		sortedSnakes := make([]game.Snake, len(state.Snakes))
		copy(sortedSnakes, state.Snakes)
		sort.Slice(sortedSnakes, func(i, j int) bool {
			return sortedSnakes[i].Id < sortedSnakes[j].Id
		})

		for _, s := range sortedSnakes {
			if s.Health <= 0 {
				continue
			}
			move, ok := moves[s.Id]
			if !ok {
				continue
			}
			localState := state.Clone()
			localState.YouId = s.Id

			xPtr := convert.StateToBytes(localState)
			x := make([]byte, len(*xPtr))
			copy(x, *xPtr)
			convert.PutBuffer(xPtr)

			rows = append(rows, store.TrainingRow{
				GameID: gameID,
				Turn:   state.Turn,
				EgoID:  s.Id,
				Width:  state.Width,
				Height: state.Height,
				X:      x,
				Policy: int32(move),
				Value:  0,
			})
		}

		if onStep != nil {
			onStep()
		}

		// Advance State Simultaneously (with food spawning randomness)
		state = rules.NextStateSimultaneousWithRNG(state, moves, rng)
	}

	// Determine Winner
	winnerId := ""
	living := 0
	for _, s := range state.Snakes {
		if s.Health > 0 {
			living++
			winnerId = s.Id
		}
	}
	if living != 1 {
		winnerId = "" // Draw or everyone died
	}

	// Assign values after outcome is known.
	if winnerId == "" {
		for i := range rows {
			rows[i].Value = 0
		}
	} else {
		for i := range rows {
			if rows[i].EgoID == winnerId {
				rows[i].Value = 1
			} else {
				rows[i].Value = -1
			}
		}
	}

	return rows, GameResult{WinnerId: winnerId, Steps: int(state.Turn)}
}

func createInitialState(rng *rand.Rand) *game.GameState {
	state := &game.GameState{
		Width:  11,
		Height: 11,
		YouId:  "snake1",
		Snakes: []game.Snake{
			{
				Id:     "snake1",
				Health: 100,
				Body: []game.Point{
					{X: 1, Y: 1},
					{X: 1, Y: 1},
					{X: 1, Y: 1},
				},
			},
			{
				Id:     "snake2",
				Health: 100,
				Body: []game.Point{
					{X: 9, Y: 9},
					{X: 9, Y: 9},
					{X: 9, Y: 9},
				},
			},
		},
		Food: nil,
		Turn: 0,
	}

	// Spawn initial food like Battlesnake: enforce minimum food at game start.
	// Use chance=0 so we only ensure the minimum.
	rules.ApplyFoodSettings(state, rng, rules.FoodSettings{MinimumFood: 1, FoodSpawnChance: 0})
	return state
}

func sampleMove(rng *rand.Rand, policy []float32) int {
	r := rng.Float32()
	sum := float32(0)
	for i, p := range policy {
		sum += p
		if r < sum {
			return i
		}
	}
	return len(policy) - 1
}

func argmax(policy []float32) int {
	bestIdx := -1
	bestVal := float32(-1)
	for i, p := range policy {
		if p > bestVal {
			bestVal = p
			bestIdx = i
		}
	}
	return bestIdx
}
