package selfplay

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"sort"
	"time"

	"github.com/brensch/snek2/executor/mcts"
	"github.com/brensch/snek2/game"
	"github.com/brensch/snek2/rules"
	"github.com/brensch/snek2/scraper/store"
)

type GameResult struct {
	WinnerId string
	Steps    int
}

// InProgressGame is a resumable self-play game snapshot.
//
// It includes the current game state plus any already-recorded per-turn archive rows.
// Values are assigned only once the game completes.
type InProgressGame struct {
	GameID   string                 `json:"game_id"`
	State    *game.GameState        `json:"state"`
	Rows     []store.ArchiveTurnRow `json:"rows"`
	RNGSeed  int64                  `json:"rng_seed"`
	PausedAt int32                  `json:"paused_at"`
}

type PlayGameOutcome struct {
	Completed  bool
	Rows       []store.ArchiveTurnRow
	Result     GameResult
	Checkpoint *InProgressGame
}

type PlayGameOptions struct {
	Resume        *InProgressGame
	StopRequested func() bool
	ModelPath     string // Resolved path to the ONNX model used
}

// PlayGame is the legacy API used by older callers. It will abort and return nil rows
// if the context is cancelled.
func PlayGame(ctx context.Context, workerId int, mctsConfig mcts.Config, client mcts.Predictor, verbose bool, sims int, onStep func()) ([]store.ArchiveTurnRow, GameResult) {
	out := PlayGameWithOptions(ctx, workerId, mctsConfig, client, verbose, sims, onStep, PlayGameOptions{})
	if !out.Completed {
		return nil, out.Result
	}
	return out.Rows, out.Result
}

func PlayGameWithOptions(ctx context.Context, workerId int, mctsConfig mcts.Config, client mcts.Predictor, verbose bool, sims int, onStep func(), opts PlayGameOptions) PlayGameOutcome {
	stopRequested := opts.StopRequested
	if stopRequested == nil {
		stopRequested = func() bool { return false }
	}

	var rngSeed int64
	var state *game.GameState
	var gameID string
	rows := make([]store.ArchiveTurnRow, 0, 256)

	if opts.Resume != nil && opts.Resume.State != nil && opts.Resume.GameID != "" {
		gameID = opts.Resume.GameID
		state = opts.Resume.State.Clone()
		rngSeed = opts.Resume.RNGSeed
		if rngSeed == 0 {
			rngSeed = time.Now().UnixNano() + int64(workerId)*1000003
		}
		if len(opts.Resume.Rows) > 0 {
			rows = append(rows, opts.Resume.Rows...)
		}
	} else {
		rngSeed = time.Now().UnixNano() + int64(workerId)*1000003
		rng := rand.New(rand.NewSource(rngSeed))
		state = createInitialState(rng)
		gameID = fmt.Sprintf("selfplay_%d_%d", time.Now().UnixNano(), workerId)
	}

	rng := rand.New(rand.NewSource(rngSeed))

	moveNames := []string{"Up", "Down", "Left", "Right"}

	for {
		if ctx != nil {
			select {
			case <-ctx.Done():
				// Gracefully checkpoint instead of throwing away the partial game.
				return PlayGameOutcome{
					Completed: false,
					Result:    GameResult{WinnerId: "", Steps: int(state.Turn)},
					Checkpoint: &InProgressGame{
						GameID:   gameID,
						State:    state.Clone(),
						Rows:     append([]store.ArchiveTurnRow(nil), rows...),
						RNGSeed:  rng.Int63(),
						PausedAt: state.Turn,
					},
				}
			default:
			}
		}

		if stopRequested() {
			return PlayGameOutcome{
				Completed: false,
				Result:    GameResult{WinnerId: "", Steps: int(state.Turn)},
				Checkpoint: &InProgressGame{
					GameID:   gameID,
					State:    state.Clone(),
					Rows:     append([]store.ArchiveTurnRow(nil), rows...),
					RNGSeed:  rng.Int63(),
					PausedAt: state.Turn,
				},
			}
		}

		if verbose {
			PrintBoard(state)
		}

		if rules.IsGameOver(state) {
			break
		}

		// Use simultaneous MCTS - alternating tree structure with shared inference per round
		// All snakes see the same state within a round, moves applied simultaneously at end
		searchSims := sims
		if searchSims <= 0 {
			searchSims = 1
		}

		simRoot, snakeOrder, err := mcts.SimultaneousSearch(ctx, client, mctsConfig, state, searchSims)
		if err != nil {
			if ctx != nil {
				select {
				case <-ctx.Done():
					return PlayGameOutcome{
						Completed: false,
						Result:    GameResult{WinnerId: "", Steps: int(state.Turn)},
						Checkpoint: &InProgressGame{
							GameID:   gameID,
							State:    state.Clone(),
							Rows:     append([]store.ArchiveTurnRow(nil), rows...),
							RNGSeed:  rng.Int63(),
							PausedAt: state.Turn,
						},
					}
				default:
				}
			}
			log.Printf("Simultaneous MCTS Error: %v", err)
			continue
		}
		_ = snakeOrder // Used internally by GetSimultaneousSearchResult

		// Extract results from simultaneous search
		searchResult := mcts.GetSimultaneousSearchResult(simRoot, snakeOrder)

		// Get moves - use sampling for early game, argmax later
		moves := make(map[string]int)
		policies := make(map[string][]float32)

		for snakeID, policy := range searchResult.Policies {
			policySlice := policy[:]
			policies[snakeID] = policySlice

			// Select move: sample early game, argmax later
			var move int
			if state.Turn < 10 {
				move = sampleMove(rng, policySlice)
			} else {
				move = argmax(policySlice)
			}
			moves[snakeID] = move
		}

		// Fallback for any snake without a policy (shouldn't happen)
		for _, snake := range state.Snakes {
			if snake.Health <= 0 {
				continue
			}
			if _, ok := moves[snake.Id]; !ok {
				moves[snake.Id] = 0 // Default up
				policies[snake.Id] = []float32{1, 0, 0, 0}
			}
		}

		if ctx != nil {
			select {
			case <-ctx.Done():
				return PlayGameOutcome{
					Completed: false,
					Result:    GameResult{WinnerId: "", Steps: int(state.Turn)},
					Checkpoint: &InProgressGame{
						GameID:   gameID,
						State:    state.Clone(),
						Rows:     append([]store.ArchiveTurnRow(nil), rows...),
						RNGSeed:  rng.Int63(),
						PausedAt: state.Turn,
					},
				}
			default:
			}
		}
		if stopRequested() {
			return PlayGameOutcome{
				Completed: false,
				Result:    GameResult{WinnerId: "", Steps: int(state.Turn)},
				Checkpoint: &InProgressGame{
					GameID:   gameID,
					State:    state.Clone(),
					Rows:     append([]store.ArchiveTurnRow(nil), rows...),
					RNGSeed:  rng.Int63(),
					PausedAt: state.Turn,
				},
			}
		}

		if verbose {
			PrintBoard(state)
			ids := make([]string, 0, len(moves))
			for id := range moves {
				ids = append(ids, id)
			}
			sort.Strings(ids)

			// Log unified search results
			log.Printf("[Worker %d] Turn %d: Unified MCTS with %d sims", workerId, state.Turn, sims)
			for _, id := range ids {
				m := moves[id]
				moveName := "Unknown"
				if m >= 0 && m < len(moveNames) {
					moveName = moveNames[m]
				}
				policy := policies[id]
				policyStr := ""
				for i, p := range policy {
					if i > 0 {
						policyStr += " "
					}
					policyStr += fmt.Sprintf("%s:%.1f%%", moveNames[i], p*100)
				}
				log.Printf("  %s -> %s | policy: %s", id, moveName, policyStr)
			}
		}

		// Record one archive row for this turn (contains all snakes).
		sortedSnakes := make([]game.Snake, len(state.Snakes))
		copy(sortedSnakes, state.Snakes)
		sort.Slice(sortedSnakes, func(i, j int) bool {
			return sortedSnakes[i].Id < sortedSnakes[j].Id
		})

		turnRow := store.ArchiveTurnRow{
			GameID:    gameID,
			Turn:      state.Turn,
			Width:     state.Width,
			Height:    state.Height,
			Source:    "selfplay",
			ModelPath: opts.ModelPath,
			FoodX:     nil,
			FoodY:     nil,
			HazardX:   nil,
			HazardY:   nil,
			Snakes:    nil,
		}

		// Serialize MCTS root children summary
		if searchResult != nil && len(searchResult.RootChildren) > 0 {
			if rootJSON, err := json.Marshal(searchResult.RootChildren); err == nil {
				turnRow.MCTSRootJSON = rootJSON
			}
		}

		if len(state.Food) > 0 {
			turnRow.FoodX = make([]int32, 0, len(state.Food))
			turnRow.FoodY = make([]int32, 0, len(state.Food))
			for _, p := range state.Food {
				turnRow.FoodX = append(turnRow.FoodX, p.X)
				turnRow.FoodY = append(turnRow.FoodY, p.Y)
			}
		}

		turnRow.Snakes = make([]store.ArchiveSnake, 0, len(sortedSnakes))
		for _, s := range sortedSnakes {
			alive := s.Health > 0 && len(s.Body) > 0
			policy := int32(-1)
			var policyProbs []float32
			if alive {
				if move, ok := moves[s.Id]; ok {
					policy = int32(move)
				}
				if p, ok := policies[s.Id]; ok {
					// Copy to avoid sharing underlying arrays.
					policyProbs = append([]float32(nil), p...)
				}
			}
			if policy >= 0 && policy < 4 && len(policyProbs) != 4 {
				policyProbs = make([]float32, 4)
				policyProbs[policy] = 1.0
			}
			snakeRow := store.ArchiveSnake{
				ID:          s.Id,
				Alive:       alive,
				Health:      s.Health,
				Policy:      policy,
				PolicyProbs: policyProbs,
				Value:       0,
			}
			if len(s.Body) > 0 {
				snakeRow.BodyX = make([]int32, 0, len(s.Body))
				snakeRow.BodyY = make([]int32, 0, len(s.Body))
				for _, bp := range s.Body {
					snakeRow.BodyX = append(snakeRow.BodyX, bp.X)
					snakeRow.BodyY = append(snakeRow.BodyY, bp.Y)
				}
			}
			turnRow.Snakes = append(turnRow.Snakes, snakeRow)
		}

		rows = append(rows, turnRow)

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

	// Record the terminal (game-over) state as a final row.
	// The per-turn loop records snapshots before applying moves; without this,
	// completed games appear to stop in a non-terminal position.
	sortedSnakes := make([]game.Snake, len(state.Snakes))
	copy(sortedSnakes, state.Snakes)
	sort.Slice(sortedSnakes, func(i, j int) bool {
		return sortedSnakes[i].Id < sortedSnakes[j].Id
	})

	terminalRow := store.ArchiveTurnRow{
		GameID:  gameID,
		Turn:    state.Turn,
		Width:   state.Width,
		Height:  state.Height,
		Source:  "selfplay",
		FoodX:   nil,
		FoodY:   nil,
		HazardX: nil,
		HazardY: nil,
		Snakes:  nil,
	}
	if len(state.Food) > 0 {
		terminalRow.FoodX = make([]int32, 0, len(state.Food))
		terminalRow.FoodY = make([]int32, 0, len(state.Food))
		for _, p := range state.Food {
			terminalRow.FoodX = append(terminalRow.FoodX, p.X)
			terminalRow.FoodY = append(terminalRow.FoodY, p.Y)
		}
	}

	terminalRow.Snakes = make([]store.ArchiveSnake, 0, len(sortedSnakes))
	for _, s := range sortedSnakes {
		alive := s.Health > 0 && len(s.Body) > 0
		snakeRow := store.ArchiveSnake{
			ID:          s.Id,
			Alive:       alive,
			Health:      s.Health,
			Policy:      -1,
			PolicyProbs: nil,
			Value:       0,
		}
		if len(s.Body) > 0 {
			snakeRow.BodyX = make([]int32, 0, len(s.Body))
			snakeRow.BodyY = make([]int32, 0, len(s.Body))
			for _, bp := range s.Body {
				snakeRow.BodyX = append(snakeRow.BodyX, bp.X)
				snakeRow.BodyY = append(snakeRow.BodyY, bp.Y)
			}
		}
		terminalRow.Snakes = append(terminalRow.Snakes, snakeRow)
	}

	rows = append(rows, terminalRow)

	// Assign values after outcome is known.
	// Apply a per-turn discount (gamma) to WINS so faster wins are worth more.
	// Losses remain at flat -1 to create asymmetric pressure favoring aggression:
	// - Passive play risks full loss penalty with diminished win reward
	// - Aggressive play has same downside but better upside
	const gamma = 0.997 // Discount factor per turn (~0.74 value at turn 100, ~0.55 at turn 200)

	totalTurns := len(rows)
	for i := range rows {
		// Discount based on how many turns remain until game end
		turnsRemaining := totalTurns - 1 - i
		discount := float32(math.Pow(float64(gamma), float64(turnsRemaining)))

		for j := range rows[i].Snakes {
			if winnerId == "" {
				// Treat draws as a negative outcome to discourage "suicidal" play
				// where both snakes die simultaneously. No discount - draws are always bad.
				rows[i].Snakes[j].Value = -0.5
				continue
			}
			if rows[i].Snakes[j].ID == winnerId {
				// Discount wins - faster wins are worth more
				rows[i].Snakes[j].Value = 1 * discount
			} else {
				// Flat loss penalty - losing is always maximally bad
				rows[i].Snakes[j].Value = -1
			}
		}
	}

	// Generate final game_id with finish time (not start time) so the viewer
	// shows when the game completed rather than when it started.
	finishedGameID := fmt.Sprintf("selfplay_%d_%d", time.Now().UnixNano(), workerId)
	for i := range rows {
		rows[i].GameID = finishedGameID
	}

	return PlayGameOutcome{Completed: true, Rows: rows, Result: GameResult{WinnerId: winnerId, Steps: int(state.Turn)}}
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
