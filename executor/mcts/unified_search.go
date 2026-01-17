package mcts

import (
	"context"
	"math"

	"github.com/brensch/snek2/game"
	"github.com/brensch/snek2/rules"
)

// JointAction represents all snakes' moves for a single game step
type JointAction struct {
	Moves map[string]int // snakeID -> move
}

// JointNode represents a node in the unified MCTS tree for simultaneous games.
// Each child represents a joint action (all snakes moving together).
type JointNode struct {
	State      *game.GameState
	VisitCount int
	ValueSum   float32 // Sum of values (from ego perspective, or could be per-snake)

	// Children keyed by encoded joint action
	Children map[string]*JointChild

	// Priors for each snake's moves (computed during expansion)
	SnakePriors map[string][4]float32

	IsExpanded bool
}

// JointChild represents a child node with its joint action
type JointChild struct {
	Action     JointAction
	Node       *JointNode
	PriorProb  float32 // Product of individual snake priors
	VisitCount int
	ValueSum   float32
}

// encodeAction creates a string key for a joint action
func encodeAction(moves map[string]int, order []string) string {
	result := ""
	for _, id := range order {
		if mv, ok := moves[id]; ok {
			result += string(rune('0' + mv))
		}
	}
	return result
}

// NewJointNode creates a new joint node
func NewJointNode(state *game.GameState) *JointNode {
	return &JointNode{
		State:    state,
		Children: make(map[string]*JointChild),
	}
}

// UnifiedSearch runs MCTS where all snakes explore simultaneously.
// This is the correct approach for self-play in simultaneous games.
func UnifiedSearch(ctx context.Context, client Predictor, config Config, rootState *game.GameState, simulations int) (*JointNode, error) {
	root := NewJointNode(rootState)

	// Get consistent snake order
	snakeOrder := make([]string, 0, len(rootState.Snakes))
	for _, s := range rootState.Snakes {
		if s.Health > 0 {
			snakeOrder = append(snakeOrder, s.Id)
		}
	}

	for i := 0; i < simulations; i++ {
		if ctx != nil {
			select {
			case <-ctx.Done():
				return root, ctx.Err()
			default:
			}
		}

		node := root
		path := []*JointNode{node}
		childPath := []*JointChild{}

		// Selection - traverse down using UCB on joint actions
		for node.IsExpanded && len(node.Children) > 0 {
			bestKey := ""
			bestScore := float32(-1e9)
			var bestChild *JointChild

			sqrtN := float32(math.Sqrt(float64(node.VisitCount)))

			for key, child := range node.Children {
				q := float32(0)
				if child.VisitCount > 0 {
					q = child.ValueSum / float32(child.VisitCount)
				}

				// UCB with joint prior
				u := q + config.Cpuct*child.PriorProb*sqrtN/(1+float32(child.VisitCount))

				if u > bestScore {
					bestScore = u
					bestKey = key
					bestChild = child
				}
			}

			if bestChild == nil {
				break
			}

			childPath = append(childPath, bestChild)
			node = bestChild.Node
			path = append(path, node)
			_ = bestKey
		}

		// Expansion & Evaluation
		value := float32(0)

		if rules.IsGameOver(node.State) {
			// Terminal - evaluate from first snake's perspective (they're all the same model)
			tempState := node.State.Clone()
			if len(snakeOrder) > 0 {
				tempState.YouId = snakeOrder[0]
			}
			value = rules.GetResult(tempState)
		} else if !node.IsExpanded {
			// Expand: get policy for each alive snake
			value = expandJointNode(ctx, client, node, snakeOrder)
			if ctx != nil && ctx.Err() != nil {
				return root, ctx.Err()
			}
		}

		// Backpropagation
		for _, n := range path {
			n.VisitCount++
			n.ValueSum += value
		}
		for _, c := range childPath {
			c.VisitCount++
			c.ValueSum += value
		}
	}

	return root, nil
}

// expandJointNode expands a node by creating children for all joint actions
func expandJointNode(ctx context.Context, client Predictor, node *JointNode, snakeOrder []string) float32 {
	// Get alive snakes at this state
	aliveSnakes := make([]string, 0)
	for _, s := range node.State.Snakes {
		if s.Health > 0 {
			aliveSnakes = append(aliveSnakes, s.Id)
		}
	}

	if len(aliveSnakes) == 0 {
		node.IsExpanded = true
		return 0
	}

	// Get policy and value for each snake
	node.SnakePriors = make(map[string][4]float32)
	values := make(map[string]float32)

	for _, snakeID := range aliveSnakes {
		stateForSnake := node.State.Clone()
		stateForSnake.YouId = snakeID

		policyLogits, vals, err := client.Predict(stateForSnake)
		if err != nil {
			continue
		}

		node.SnakePriors[snakeID] = softmax4(policyLogits)
		if len(vals) > 0 {
			values[snakeID] = vals[0]
		}
	}

	// Use the first snake's value as the node value (in self-play, all snakes are equivalent)
	avgValue := float32(0)
	for _, v := range values {
		avgValue += v
	}
	if len(values) > 0 {
		avgValue /= float32(len(values))
	}

	// Get legal moves for each snake.
	// Use GetLegalMovesWithTailDecrement so snakes can see that tail positions will be free.
	// Each snake evaluates on the same base state (tails decremented, but no other snake moves committed).
	legalMoves := make(map[string][]int)
	for _, snakeID := range aliveSnakes {
		stateForSnake := node.State.Clone()
		stateForSnake.YouId = snakeID
		moves := rules.GetLegalMovesWithTailDecrement(stateForSnake)
		if len(moves) == 0 {
			moves = []int{0} // default if no legal moves
		}
		legalMoves[snakeID] = moves
	}

	// Generate all joint actions (cartesian product of legal moves)
	// For efficiency, limit to top moves by prior if too many combinations
	jointActions := generateJointActions(aliveSnakes, legalMoves, node.SnakePriors)

	for _, action := range jointActions {
		if ctx != nil {
			select {
			case <-ctx.Done():
				return avgValue
			default:
			}
		}

		// Calculate joint prior (product of individual priors)
		jointPrior := float32(1.0)
		for snakeID, move := range action.Moves {
			if priors, ok := node.SnakePriors[snakeID]; ok {
				jointPrior *= priors[move]
			}
		}

		// Simulate the joint action
		nextState := rules.NextStateSimultaneous(node.State, action.Moves)

		child := &JointChild{
			Action:    action,
			Node:      NewJointNode(nextState),
			PriorProb: jointPrior,
		}

		key := encodeAction(action.Moves, aliveSnakes)
		node.Children[key] = child
	}

	node.IsExpanded = true
	return avgValue
}

// generateJointActions generates all combinations of legal moves
// For efficiency, can limit to top-k moves per snake if needed
func generateJointActions(snakes []string, legalMoves map[string][]int, priors map[string][4]float32) []JointAction {
	if len(snakes) == 0 {
		return []JointAction{{Moves: make(map[string]int)}}
	}

	// Recursive: get actions for remaining snakes, then prepend this snake's moves
	firstSnake := snakes[0]
	restSnakes := snakes[1:]
	restActions := generateJointActions(restSnakes, legalMoves, priors)

	moves := legalMoves[firstSnake]

	// Limit moves if too many (keep top 2-3 by prior)
	if len(moves) > 3 && len(snakes) > 2 {
		snakePriors := priors[firstSnake]
		type movePrior struct {
			move int
			p    float32
		}
		mps := make([]movePrior, len(moves))
		for i, m := range moves {
			mps[i] = movePrior{m, snakePriors[m]}
		}
		// Sort by prior descending
		for i := 0; i < len(mps)-1; i++ {
			for j := i + 1; j < len(mps); j++ {
				if mps[j].p > mps[i].p {
					mps[i], mps[j] = mps[j], mps[i]
				}
			}
		}
		moves = make([]int, 0, 3)
		for i := 0; i < 3 && i < len(mps); i++ {
			moves = append(moves, mps[i].move)
		}
	}

	result := make([]JointAction, 0, len(moves)*len(restActions))
	for _, move := range moves {
		for _, restAction := range restActions {
			newMoves := make(map[string]int, len(restAction.Moves)+1)
			for k, v := range restAction.Moves {
				newMoves[k] = v
			}
			newMoves[firstSnake] = move
			result = append(result, JointAction{Moves: newMoves})
		}
	}

	return result
}

// GetBestMoves returns the best move for each snake from the unified tree
func GetBestMoves(root *JointNode) map[string]int {
	if root == nil || len(root.Children) == 0 {
		return nil
	}

	// Find the most visited child
	var bestChild *JointChild
	bestN := -1
	for _, child := range root.Children {
		if child.VisitCount > bestN {
			bestN = child.VisitCount
			bestChild = child
		}
	}

	if bestChild == nil {
		return nil
	}

	return bestChild.Action.Moves
}

// JointSearchResult contains the results of a unified MCTS search
type JointSearchResult struct {
	Root *JointNode

	// Per-snake derived statistics
	Moves    map[string]int        // Best move for each snake
	Policies map[string][4]float32 // Visit distribution per snake
	Values   map[string]float32    // Q-value of best joint action per snake

	// Root children summary (for storage without full tree)
	RootChildren []JointChildSummary
}

// JointChildSummary is a compact representation of a root child for storage
type JointChildSummary struct {
	Moves      map[string]int `json:"moves"`
	VisitCount int            `json:"n"`
	ValueSum   float32        `json:"value_sum"`
	Q          float32        `json:"q"`
	PriorProb  float32        `json:"p"`
}

// GetSearchResult extracts all useful data from the search for both gameplay and storage
func GetSearchResult(root *JointNode) *JointSearchResult {
	if root == nil {
		return nil
	}

	result := &JointSearchResult{
		Root:         root,
		Moves:        make(map[string]int),
		Policies:     make(map[string][4]float32),
		Values:       make(map[string]float32),
		RootChildren: make([]JointChildSummary, 0, len(root.Children)),
	}

	if len(root.Children) == 0 {
		return result
	}

	// Collect visit counts per snake per move
	snakeVisits := make(map[string][4]int)
	totalVisits := 0

	var bestChild *JointChild
	bestN := -1

	for _, child := range root.Children {
		totalVisits += child.VisitCount

		if child.VisitCount > bestN {
			bestN = child.VisitCount
			bestChild = child
		}

		// Accumulate per-snake visit counts
		for snakeID, move := range child.Action.Moves {
			visits := snakeVisits[snakeID]
			visits[move] += child.VisitCount
			snakeVisits[snakeID] = visits
		}

		// Build root children summary
		q := float32(0)
		if child.VisitCount > 0 {
			q = child.ValueSum / float32(child.VisitCount)
		}
		result.RootChildren = append(result.RootChildren, JointChildSummary{
			Moves:      child.Action.Moves,
			VisitCount: child.VisitCount,
			ValueSum:   child.ValueSum,
			Q:          q,
			PriorProb:  child.PriorProb,
		})
	}

	// Set best moves
	if bestChild != nil {
		result.Moves = bestChild.Action.Moves

		// Value is the Q of the best joint action
		if bestChild.VisitCount > 0 {
			q := bestChild.ValueSum / float32(bestChild.VisitCount)
			for snakeID := range bestChild.Action.Moves {
				result.Values[snakeID] = q
			}
		}
	}

	// Compute per-snake policies (normalized visit counts)
	for snakeID, visits := range snakeVisits {
		total := 0
		for _, v := range visits {
			total += v
		}
		if total > 0 {
			var policy [4]float32
			for i, v := range visits {
				policy[i] = float32(v) / float32(total)
			}
			result.Policies[snakeID] = policy
		}
	}

	return result
}
