// Package mcts tests - verifies SimultaneousSearch for multi-snake MCTS.
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

func TestSimultaneousSearch(t *testing.T) {
	// Setup
	client := &MockInferenceClient{}
	config := Config{Cpuct: 1.0}

	// Create a simple game state with two snakes
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
			{
				Id:     "enemy",
				Health: 100,
				Body: []game.Point{
					{X: 8, Y: 8},
					{X: 8, Y: 7},
					{X: 8, Y: 6},
				},
			},
		},
		Food: []game.Point{{X: 2, Y: 2}},
	}

	// Run SimultaneousSearch
	simulations := 10
	root, snakeOrder, err := SimultaneousSearch(context.Background(), client, config, state, simulations)
	if err != nil {
		t.Fatalf("SimultaneousSearch failed: %v", err)
	}

	// Assertions
	if root.VisitCount != simulations {
		t.Errorf("Expected VisitCount %d, got %d", simulations, root.VisitCount)
	}

	if len(snakeOrder) != 2 {
		t.Errorf("Expected 2 snakes in order, got %d", len(snakeOrder))
	}

	// Check if children have visits
	childrenFound := 0
	for _, child := range root.Children {
		if child != nil {
			childrenFound++
		}
	}

	if childrenFound == 0 {
		t.Errorf("Expected children, got none")
	}

	// Extract results and verify policies exist
	result := GetSimultaneousSearchResult(root, snakeOrder)
	if len(result.Policies) == 0 {
		t.Error("Expected policies in result")
	}

	// Check that we got a policy for the first snake
	if _, ok := result.Policies["me"]; !ok {
		t.Error("Expected policy for 'me' snake")
	}
}

func BenchmarkSimultaneousSearch(b *testing.B) {
	// Setup
	client := &MockInferenceClient{}
	config := Config{Cpuct: 1.0}

	// Create a simple game state with two snakes
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
			{
				Id:     "enemy",
				Health: 100,
				Body: []game.Point{
					{X: 8, Y: 8},
					{X: 8, Y: 7},
					{X: 8, Y: 6},
				},
			},
		},
		Food: []game.Point{{X: 2, Y: 2}},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, err := SimultaneousSearch(context.Background(), client, config, state, 100)
		if err != nil {
			b.Fatalf("SimultaneousSearch failed: %v", err)
		}
	}
}
