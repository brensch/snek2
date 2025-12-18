package selfplay

import (
	"log"
	"math/rand"
	"time"

	pb "github.com/brensch/snek2/gen/go"
	"github.com/brensch/snek2/go-worker/convert"
	"github.com/brensch/snek2/go-worker/mcts"
	"github.com/brensch/snek2/rules"
)

type GameResult struct {
	WinnerId string
	Steps    int
}

func PlayGame(workerId int, mctsConfig mcts.Config, client pb.InferenceServiceClient, verbose bool) ([]*pb.TrainingExample, GameResult) {
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	state := createInitialState()

	var examples []*pb.TrainingExample

	type step struct {
		stateData []byte
		policy    []float32
		playerId  string
	}
	var steps []step

	moveNames := []string{"Up", "Down", "Left", "Right"}

	for {
		if verbose {
			PrintBoard(state)
		}

		if rules.IsGameOver(state) {
			break
		}

		moves := make(map[string]int)
		
		// Iterate over all snakes to get their moves
		// We use a loop index to avoid issues if state.Snakes changes order (though it shouldn't)
		for _, snake := range state.Snakes {
			if snake.Health <= 0 {
				continue
			}

			// Set perspective for MCTS and Feature Extraction
			state.YouId = snake.Id

			// MCTS Search
			mctsInstance := &mcts.MCTS{
				Config: mctsConfig,
				Client: client,
			}
			root, err := mctsInstance.Search(state, 800)
			if err != nil {
				// If inference fails, abort game
				return nil, GameResult{WinnerId: "Error", Steps: int(state.Turn)}
			}

			// Extract Policy
			policy := make([]float32, 4)
			totalVisits := 0
			for _, child := range root.Children {
				totalVisits += child.VisitCount
			}

			// If no valid moves found (e.g. trapped), pick a default
			if totalVisits == 0 {
				moves[snake.Id] = 0 // Default Up
				continue
			}

			for move, child := range root.Children {
				if int(move) < len(policy) {
					policy[int(move)] = float32(child.VisitCount) / float32(totalVisits)
				}
			}

			// Select Move
			var move int
			if state.Turn < 10 {
				move = sampleMove(rng, policy)
			} else {
				move = argmax(policy)
			}
			moves[snake.Id] = move

			if verbose {
				moveName := "Unknown"
				if move >= 0 && move < len(moveNames) {
					moveName = moveNames[move]
				}
				log.Printf("[Worker %d] Turn %d: %s chose %s (Visits: %d)", workerId, state.Turn, snake.Id, moveName, totalVisits)
			}

			// Record Step
			stateBytesPtr := convert.StateToBytes(state)
			stateBytes := make([]byte, len(*stateBytesPtr))
			copy(stateBytes, *stateBytesPtr)
			convert.PutBuffer(stateBytesPtr)

			steps = append(steps, step{
				stateData: stateBytes,
				policy:    policy,
				playerId:  snake.Id,
			})
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
		value := float32(0)
		if winnerId != "" {
			if s.playerId == winnerId {
				value = 1.0
			} else {
				value = -1.0
			}
		}
		
		examples = append(examples, &pb.TrainingExample{
			StateData:    s.stateData,
			PolicyTarget: s.policy,
			ValueTarget:  value,
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
					{X: 9, Y: 8},
					{X: 9, Y: 7},
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
