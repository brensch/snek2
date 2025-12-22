package mcts

import (
	"github.com/brensch/snek2/game"
)

// Move represents a direction (0: Up, 1: Down, 2: Left, 3: Right)
type Move int

// Node represents a state in the MCTS tree
type Node struct {
	VisitCount int
	ValueSum   float32
	PriorProb  float32
	Children   [4]*Node
	State      *game.GameState
	IsExpanded bool
}

// NewNode creates a new MCTS node
func NewNode(state *game.GameState, prior float32) *Node {
	return &Node{
		State:     state,
		PriorProb: prior,
	}
}

// Config holds MCTS configuration
type Config struct {
	Cpuct float32
}

// Predictor defines the interface for inference
type Predictor interface {
	Predict(state *game.GameState) ([]float32, []float32, error)
}

// MCTS holds the search context
type MCTS struct {
	Config Config
	Client Predictor
}
