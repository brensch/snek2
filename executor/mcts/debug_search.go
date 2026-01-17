package mcts

import (
	"context"
	"math"

	"github.com/brensch/snek2/game"
	"github.com/brensch/snek2/rules"
)

// DebugNode represents a node in the alternating MCTS tree.
// The tree alternates between snakes - each node represents one snake choosing a move.
// After all alive snakes have chosen, the game state is resolved.
type DebugNode struct {
	VisitCount int
	ValueSum   float32
	PriorProb  float32
	Children   [4]*DebugNode

	// SnakeID is the snake that made the move leading to this node
	SnakeID string
	// Move is the move that was made (0=Up, 1=Down, 2=Left, 3=Right)
	Move int

	// State is the game state at this node.
	// For intermediate nodes (not all snakes have moved yet), this is the same as the parent.
	// For "resolved" nodes (all snakes have moved), this is the new state after moves are applied.
	State *game.GameState

	// PendingMoves tracks moves made so far this "round" (before state resolution)
	PendingMoves map[string]int

	// NextSnakeIdx is the index of the next snake to move (in the snake order)
	// -1 means this is a "resolved" node and the state has been updated
	NextSnakeIdx int

	// SnakeOrder is the order in which snakes take turns
	SnakeOrder []string

	IsExpanded bool
}

// NewDebugRootNode creates a root node for debug MCTS.
func NewDebugRootNode(state *game.GameState) *DebugNode {
	// Build snake order from alive snakes
	order := make([]string, 0, len(state.Snakes))
	for _, s := range state.Snakes {
		if s.Health > 0 {
			order = append(order, s.Id)
		}
	}

	return &DebugNode{
		State:        state,
		PendingMoves: make(map[string]int),
		NextSnakeIdx: 0,
		SnakeOrder:   order,
		PriorProb:    1.0,
		Move:         -1, // root has no move
	}
}

// DebugSearch runs MCTS with alternating snake moves for debugging.
// This creates a shared tree where each level represents one snake's decision.
func DebugSearch(ctx context.Context, client Predictor, config Config, rootState *game.GameState, simulations int) (*DebugNode, error) {
	root := NewDebugRootNode(rootState)

	for i := 0; i < simulations; i++ {
		if ctx != nil {
			select {
			case <-ctx.Done():
				return root, ctx.Err()
			default:
			}
		}

		node := root
		path := []*DebugNode{node}

		// Selection - traverse down the tree
		for node.IsExpanded && len(node.Children) > 0 {
			bestMove := -1
			bestScore := float32(-1e9)

			sqrtSumN := float32(math.Sqrt(float64(node.VisitCount)))

			for moveIdx := 0; moveIdx < 4; moveIdx++ {
				child := node.Children[moveIdx]
				if child == nil {
					continue
				}

				q := float32(0)
				if child.VisitCount > 0 {
					q = child.ValueSum / float32(child.VisitCount)
				}

				u := q + config.Cpuct*child.PriorProb*sqrtSumN/(1+float32(child.VisitCount))

				if u > bestScore {
					bestScore = u
					bestMove = moveIdx
				}
			}

			if bestMove < 0 {
				break
			}

			node = node.Children[bestMove]
			path = append(path, node)
		}

		// Expansion & Evaluation
		value := float32(0)

		if rules.IsGameOver(node.State) {
			// Terminal node - get result from perspective of the snake that just moved
			tempState := node.State.Clone()
			if node.SnakeID != "" {
				tempState.YouId = node.SnakeID
			} else if len(node.SnakeOrder) > 0 {
				tempState.YouId = node.SnakeOrder[0]
			}
			value = rules.GetResult(tempState)
		} else if !node.IsExpanded {
			// Expand this node
			value = expandDebugNode(ctx, client, node, config)
			if ctx != nil && ctx.Err() != nil {
				return root, ctx.Err()
			}
		}

		// Backpropagation
		// Note: we need to flip the value for alternating players
		for i := len(path) - 1; i >= 0; i-- {
			n := path[i]
			n.VisitCount++
			// Value is from the perspective of the snake that moved at this node
			// If the current node's snake is different from the leaf's snake, flip the value
			if n.SnakeID != "" && len(path) > 0 {
				leafSnake := path[len(path)-1].SnakeID
				if leafSnake == "" && len(path[len(path)-1].SnakeOrder) > 0 {
					leafSnake = path[len(path)-1].SnakeOrder[0]
				}
				if n.SnakeID != leafSnake {
					n.ValueSum -= value // opponent's gain is our loss
				} else {
					n.ValueSum += value
				}
			} else {
				n.ValueSum += value
			}
		}
	}

	return root, nil
}

// expandDebugNode expands a node by adding children for the next snake's moves.
func expandDebugNode(ctx context.Context, client Predictor, node *DebugNode, config Config) float32 {
	if len(node.SnakeOrder) == 0 {
		return 0
	}

	// Determine which snake is moving
	snakeIdx := node.NextSnakeIdx
	if snakeIdx < 0 || snakeIdx >= len(node.SnakeOrder) {
		// All snakes have moved, this shouldn't happen during expansion
		return 0
	}

	snakeID := node.SnakeOrder[snakeIdx]

	// Create state from this snake's perspective for inference
	stateForInference := node.State.Clone()
	stateForInference.YouId = snakeID

	// Get legal moves
	legalMoves := rules.GetLegalMoves(stateForInference)
	if len(legalMoves) == 0 {
		// No legal moves - snake is trapped, will die
		legalMoves = []int{0} // default to Up, it will die anyway
	}

	// Run inference
	policyLogits, values, err := client.Predict(stateForInference)
	if err != nil {
		return 0
	}

	value := float32(0)
	if len(values) > 0 {
		value = values[0]
	}

	priors := softmax4(policyLogits)

	// Is this the last snake to move this round?
	isLastSnake := snakeIdx == len(node.SnakeOrder)-1

	for _, moveInt := range legalMoves {
		if ctx != nil {
			select {
			case <-ctx.Done():
				return value
			default:
			}
		}

		prob := priors[moveInt]

		// Copy pending moves and add this snake's move
		newPending := make(map[string]int, len(node.PendingMoves)+1)
		for k, v := range node.PendingMoves {
			newPending[k] = v
		}
		newPending[snakeID] = moveInt

		var childState *game.GameState
		var nextSnakeIdx int

		if isLastSnake {
			// All snakes have moved - resolve the state
			childState = rules.NextStateSimultaneous(node.State, newPending)
			nextSnakeIdx = 0 // restart from first snake

			// Rebuild snake order for the new state (some snakes may have died)
			newOrder := make([]string, 0, len(childState.Snakes))
			for _, s := range childState.Snakes {
				if s.Health > 0 {
					newOrder = append(newOrder, s.Id)
				}
			}

			child := &DebugNode{
				VisitCount:   0,
				ValueSum:     0,
				PriorProb:    prob,
				State:        childState,
				SnakeID:      snakeID,
				Move:         moveInt,
				PendingMoves: make(map[string]int), // reset for new round
				NextSnakeIdx: nextSnakeIdx,
				SnakeOrder:   newOrder,
			}
			node.Children[moveInt] = child
		} else {
			// More snakes need to move - state stays the same
			nextSnakeIdx = snakeIdx + 1

			child := &DebugNode{
				VisitCount:   0,
				ValueSum:     0,
				PriorProb:    prob,
				State:        node.State, // same state until all snakes move
				SnakeID:      snakeID,
				Move:         moveInt,
				PendingMoves: newPending,
				NextSnakeIdx: nextSnakeIdx,
				SnakeOrder:   node.SnakeOrder,
			}
			node.Children[moveInt] = child
		}
	}

	node.IsExpanded = true
	return value
}
