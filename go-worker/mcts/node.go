package mcts

import (
	"sync"

	pb "github.com/brensch/snek2/gen/go"
)

// Move represents a direction (0: Up, 1: Down, 2: Left, 3: Right)
type Move int

// Node represents a state in the MCTS tree
type Node struct {
	mu         sync.Mutex
	VisitCount int
	ValueSum   float32
	PriorProb  float32
	Children   map[Move]*Node
	State      *pb.GameState
}

// NewNode creates a new MCTS node
func NewNode(state *pb.GameState, prior float32) *Node {
	return &Node{
		State:     state,
		PriorProb: prior,
		Children:  make(map[Move]*Node),
	}
}

// Config holds MCTS configuration
type Config struct {
	Cpuct float32
}

// MCTS holds the search context
type MCTS struct {
	Config Config
	Client pb.InferenceServiceClient
}
