package selfplay

import (
	"log"
	"math/rand"
	"sort"
	"sync"
	"time"

	pb "github.com/brensch/snek2/gen/go"
	"github.com/brensch/snek2/go-worker/convert"
	"github.com/brensch/snek2/go-worker/mcts"
	"github.com/brensch/snek2/rules"
	"google.golang.org/protobuf/proto"
)

type GameResult struct {
	WinnerId string
	Steps    int
}

func PlayGame(workerId int, mctsConfig mcts.Config, client mcts.Predictor, verbose bool, onStep func()) ([]*pb.TrainingExample, GameResult) {
	// rng := rand.New(rand.NewSource(time.Now().UnixNano())) // Unused
	state := createInitialState()

	var examples []*pb.TrainingExample

	type step struct {
		stateData []byte
		policies  []float32
		snakes    []string
	}
	var steps []step
	// var stepsMu sync.Mutex // Not needed if we append in main thread

	moveNames := []string{"Up", "Down", "Left", "Right"}

	for {
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

			wg.Add(1)
			go func(s *pb.Snake) {
				defer wg.Done()

				// Deep copy state for this snake's perspective
				// Important because we modify YouId
				localState := proto.Clone(state).(*pb.GameState)
				localState.YouId = s.Id

				// MCTS Search
				mctsInstance := &mcts.MCTS{
					Config: mctsConfig,
					Client: client,
				}
				root, depth, err := mctsInstance.Search(localState, 800)
				if err != nil {
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
					moves[s.Id] = 0 // Default Up
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
				moves[s.Id] = move
				movesMu.Unlock()

				policiesMu.Lock()
				policies[s.Id] = policy
				policiesMu.Unlock()

				if verbose {
					moveName := "Unknown"
					if move >= 0 && move < len(moveNames) {
						moveName = moveNames[move]
					}
					log.Printf("[Worker %d] Turn %d: %s chose %s (Visits: %d, Depth: %d)", workerId, localState.Turn, s.Id, moveName, totalVisits, depth)
				}
			}(snake)
		}
		wg.Wait()

		// Record Step (Global)
		sortedSnakes := make([]*pb.Snake, len(state.Snakes))
		copy(sortedSnakes, state.Snakes)
		sort.Slice(sortedSnakes, func(i, j int) bool {
			return sortedSnakes[i].Id < sortedSnakes[j].Id
		})

		stepPolicies := make([]float32, 16)
		stepSnakeIDs := make([]string, 4)

		for i := 0; i < 4 && i < len(sortedSnakes); i++ {
			s := sortedSnakes[i]
			stepSnakeIDs[i] = s.Id
			if p, ok := policies[s.Id]; ok {
				copy(stepPolicies[i*4:], p)
			}
		}

		stateBytesPtr := convert.StateToBytes(state)
		stateBytes := make([]byte, len(*stateBytesPtr))
		copy(stateBytes, *stateBytesPtr)
		convert.PutBuffer(stateBytesPtr)

		steps = append(steps, step{
			stateData: stateBytes,
			policies:  stepPolicies,
			snakes:    stepSnakeIDs,
		})

		if onStep != nil {
			onStep()
		}

		// Advance State Simultaneously
		state = rules.NextStateSimultaneous(state, moves)
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

	// Assign Values
	for _, s := range steps {
		stepValues := make([]float32, 4)
		for i, id := range s.snakes {
			if id == "" {
				continue
			}
			if winnerId != "" {
				if id == winnerId {
					stepValues[i] = 1.0
				} else {
					stepValues[i] = -1.0
				}
			}
		}

		examples = append(examples, &pb.TrainingExample{
			StateData: s.stateData,
			Policies:  s.policies,
			Values:    stepValues,
		})
	}

	return examples, GameResult{WinnerId: winnerId, Steps: int(state.Turn)}
}

func createInitialState() *pb.GameState {
	return &pb.GameState{
		Width:  11,
		Height: 11,
		YouId:  "snake1",
		Snakes: []*pb.Snake{
			{
				Id:     "snake1",
				Health: 100,
				Body: []*pb.Point{
					{X: 1, Y: 1},
					{X: 1, Y: 2},
					{X: 1, Y: 3},
				},
			},
			{
				Id:     "snake2",
				Health: 100,
				Body: []*pb.Point{
					{X: 9, Y: 9},
					{X: 9, Y: 10},
					{X: 9, Y: 11},
				},
			},
		},
		Food: []*pb.Point{
			{X: 5, Y: 5},
		},
		Turn: 0,
	}
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
