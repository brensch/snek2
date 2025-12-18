package mcts

import (
	"context"
	"testing"

	pb "github.com/brensch/snek2/gen/go"
	"google.golang.org/grpc"
)

// MockInferenceClient mocks the InferenceServiceClient
type MockInferenceClient struct{}

func (m *MockInferenceClient) Predict(ctx context.Context, in *pb.InferenceRequest, opts ...grpc.CallOption) (*pb.BatchInferenceResponse, error) {
	// Return a dummy response
	// Policy: Uniform probability for 4 moves (0.25 each)
	// Value: 0.5 (neutral/winning)
	return &pb.BatchInferenceResponse{
		Responses: []*pb.InferenceResponse{
			{
				Policy: []float32{0.25, 0.25, 0.25, 0.25},
				Value:  0.5,
			},
		},
	}, nil
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
	root, err := mcts.Search(state, simulations)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}

	// Assertions
	if root.VisitCount != simulations {
		t.Errorf("Expected VisitCount %d, got %d", simulations, root.VisitCount)
	}

	if len(root.Children) == 0 {
		t.Errorf("Expected children, got none")
	}

	// Check if children have visits
	totalChildVisits := 0
	for _, child := range root.Children {
		totalChildVisits += child.VisitCount
	}
	
	if totalChildVisits != simulations-1 {
		t.Errorf("Expected sum of child visits %d, got %d", simulations-1, totalChildVisits)
	}
}
