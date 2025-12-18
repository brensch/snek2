package mcts

import (
	"testing"

	pb "github.com/brensch/snek2/gen/go"
)

// MockInferenceClient mocks the Predictor interface
type MockInferenceClient struct{}

func (m *MockInferenceClient) Predict(state *pb.GameState) ([]float32, []float32, error) {
	// Return a dummy response
	// Policies: 16 floats (4 snakes * 4 moves)
	// Values: 4 floats (1 value per snake)
	policies := make([]float32, 16)
	for i := range policies {
		policies[i] = 0.25
	}
	values := []float32{0.5, 0.5, 0.5, 0.5}

	return policies, values, nil
}

func TestSearch(t *testing.T) {
	// Setup
	client := &MockInferenceClient{}
	config := Config{Cpuct: 1.0}
	mcts := MCTS{Config: config, Client: client}

	// Create a simple game state
	state := &pb.GameState{
		Width:  11,
		Height: 11,
		YouId:  "me",
		Snakes: []*pb.Snake{
			{
				Id:     "me",
				Health: 100,
				Body: []*pb.Point{
					{X: 5, Y: 5},
					{X: 5, Y: 4},
					{X: 5, Y: 3},
				},
			},
		},
		Food: []*pb.Point{{X: 8, Y: 8}},
	}

	// Run Search
	simulations := 10
	root, _, err := mcts.Search(state, simulations)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	// Assertions
	if root.VisitCount != simulations {
		t.Errorf("Expected VisitCount %d, got %d", simulations, root.VisitCount)
	}

	// Check if children have visits
	totalChildVisits := 0
	childrenFound := 0
	for _, child := range root.Children {
		if child != nil {
			childrenFound++
			totalChildVisits += child.VisitCount
		}
	}

	if childrenFound == 0 {
		t.Errorf("Expected children, got none")
	}

	if totalChildVisits != simulations-1 {
		t.Errorf("Expected sum of child visits %d, got %d", simulations-1, totalChildVisits)
	}
}

func BenchmarkSearch(b *testing.B) {
	// Setup
	client := &MockInferenceClient{}
	config := Config{Cpuct: 1.0}
	mcts := MCTS{Config: config, Client: client}

	// Create a simple game state
	state := &pb.GameState{
		Width:  11,
		Height: 11,
		YouId:  "me",
		Snakes: []*pb.Snake{
			{
				Id:     "me",
				Health: 100,
				Body: []*pb.Point{
					{X: 5, Y: 5},
					{X: 5, Y: 4},
					{X: 5, Y: 3},
				},
			},
		},
		Food: []*pb.Point{{X: 8, Y: 8}},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, err := mcts.Search(state, 800) // 800 simulations per search
		if err != nil {
			b.Fatalf("Search failed: %v", err)
		}
	}
}
