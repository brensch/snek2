package selfplay

import (
	"context"
	"fmt"
	"log"
	"math/rand"
	"sort"
	"strings"
	"sync"
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

		moves := make(map[string]int)
		var movesMu sync.Mutex

		policies := make(map[string][]float32)
		var policiesMu sync.Mutex

		type mctsStats struct {
			ChosenMove  int
			TotalVisits int
			Visits      [4]int
			Q           [4]float32
			Prior       [4]float32
		}
		statsBySnake := make(map[string]mctsStats)
		var statsMu sync.Mutex

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

				// MCTS Search with local RNG for enemy move sampling
				mctsRng := rand.New(rand.NewSource(time.Now().UnixNano()))
				mctsInstance := &mcts.MCTS{
					Config: mctsConfig,
					Client: client,
					Rng:    mctsRng,
				}
				searchSims := sims
				if searchSims <= 0 {
					searchSims = 1
				}
				root, _, err := mctsInstance.Search(ctx, localState, searchSims)
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
				var visits [4]int
				var qs [4]float32
				var priors [4]float32
				for _, child := range root.Children {
					if child != nil {
						totalVisits += child.VisitCount
					}
				}

				// If no valid moves found (e.g. trapped), pick a default
				if totalVisits == 0 {
					oneHot := []float32{1, 0, 0, 0}
					movesMu.Lock()
					moves[id] = 0 // Default Up
					movesMu.Unlock()

					policiesMu.Lock()
					policies[id] = oneHot
					policiesMu.Unlock()
					return
				}

				numChildren := 0
				for move, child := range root.Children {
					if child != nil && int(move) < len(policy) {
						numChildren++
						idx := int(move)
						visits[idx] = child.VisitCount
						priors[idx] = child.PriorProb
						if child.VisitCount > 0 {
							qs[idx] = child.ValueSum / float32(child.VisitCount)
						}
						policy[idx] = float32(child.VisitCount) / float32(totalVisits)
					}
				}

				// Debug: Log if we have a 100% policy with multiple legal moves
				maxProb := float32(0)
				for _, p := range policy {
					if p > maxProb {
						maxProb = p
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

				statsMu.Lock()
				statsBySnake[id] = mctsStats{
					ChosenMove:  move,
					TotalVisits: totalVisits,
					Visits:      visits,
					Q:           qs,
					Prior:       priors,
				}
				statsMu.Unlock()

				if verbose {
					// Per-snake verbose logging is handled after all moves are chosen.
				}
			}(snakeID)
		}
		wg.Wait()

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
			for _, id := range ids {
				m := moves[id]
				moveName := "Unknown"
				if m >= 0 && m < len(moveNames) {
					moveName = moveNames[m]
				}
				statsMu.Lock()
				st, ok := statsBySnake[id]
				statsMu.Unlock()
				if ok {
					bestIdx := 0
					bestN := -1
					for i := 0; i < 4; i++ {
						if st.Visits[i] > bestN {
							bestN = st.Visits[i]
							bestIdx = i
						}
					}

					chosenN := 0
					chosenQ := float32(0)
					chosenP := float32(0)
					if m >= 0 && m < 4 {
						chosenN = st.Visits[m]
						chosenQ = st.Q[m]
						chosenP = st.Prior[m]
					}

					// Human-readable per-move breakdown.
					per := make([]string, 0, 4)
					for i := 0; i < 4; i++ {
						pct := float32(0)
						if st.TotalVisits > 0 {
							pct = float32(st.Visits[i]) / float32(st.TotalVisits) * 100
						}
						mark := ""
						if i == bestIdx {
							mark = "*" // most visited
						}
						per = append(per, fmt.Sprintf("%s%s: N=%d (%.1f%%) Q=%.3f P=%.3f", mark, moveNames[i], st.Visits[i], pct, st.Q[i], st.Prior[i]))
					}

					log.Printf(
						"[Worker %d] Turn %d: %s -> %s | chosen N=%d/%d Q=%.3f P=%.3f | %s",
						workerId,
						state.Turn,
						id,
						moveName,
						chosenN,
						st.TotalVisits,
						chosenQ,
						chosenP,
						strings.Join(per, " | "),
					)
				} else {
					log.Printf("[Worker %d] Turn %d: %s -> %s", workerId, state.Turn, id, moveName)
				}
			}
		}

		// Record one archive row for this turn (contains all snakes).
		sortedSnakes := make([]game.Snake, len(state.Snakes))
		copy(sortedSnakes, state.Snakes)
		sort.Slice(sortedSnakes, func(i, j int) bool {
			return sortedSnakes[i].Id < sortedSnakes[j].Id
		})

		turnRow := store.ArchiveTurnRow{
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
	for i := range rows {
		for j := range rows[i].Snakes {
			if winnerId == "" {
				// Treat draws as a negative outcome to discourage "suicidal" play
				// where both snakes die simultaneously.
				rows[i].Snakes[j].Value = -0.5
				continue
			}
			if rows[i].Snakes[j].ID == winnerId {
				rows[i].Snakes[j].Value = 1
			} else {
				rows[i].Snakes[j].Value = -1
			}
		}
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
