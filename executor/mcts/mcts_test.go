package mcts

import (
	"context"
	"testing"

	"github.com/brensch/snek2/game"
)

// MockInferenceClient mocks the Predictor interface
type MockInferenceClient struct{}

func (m *MockInferenceClient) Predict(state *game.GameState) ([]float32, []float32, error) {
	// Return a dummy response
	// Policy logits: 4 floats (Up/Down/Left/Right)
	// Values: single scalar
	policies := []float32{0, 0, 0, 0}
	values := []float32{0.5}

	return policies, values, nil
}

func TestSearch(t *testing.T) {
	// Setup
	client := &MockInferenceClient{}
	config := Config{Cpuct: 1.0}
	mcts := MCTS{Config: config, Client: client}

	// Create a simple game state
	state := &game.GameState{
		Width:  11,
		Height: 11,
		YouId:  "me",
		Snakes: []game.Snake{
			{
				Id:     "me",
				Health: 100,
				Body: []game.Point{
					{X: 5, Y: 5},
					{X: 5, Y: 4},
					{X: 5, Y: 3},
				},
			},
		},
		Food: []game.Point{{X: 8, Y: 8}},
	}

	// Run Search
	simulations := 10
	root, _, err := mcts.Search(context.Background(), state, simulations)
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
	state := &game.GameState{
		Width:  11,
		Height: 11,
		YouId:  "me",
		Snakes: []game.Snake{
			{
				Id:     "me",
				Health: 100,
				Body: []game.Point{
					{X: 5, Y: 5},
					{X: 5, Y: 4},
					{X: 5, Y: 3},
				},
			},
		},
		Food: []game.Point{{X: 8, Y: 8}},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, err := mcts.Search(context.Background(), state, 800) // 800 simulations per search
		if err != nil {
			b.Fatalf("Search failed: %v", err)
		}
	}
}
