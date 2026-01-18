package mcts

import (
	"context"
	"math"

	"github.com/brensch/snek2/game"
	"github.com/brensch/snek2/rules"
)

// AlternatingNode represents a node in the alternating MCTS tree.
// Each level corresponds to one snake's decision.
// After all snakes have chosen, the game state advances.
type AlternatingNode struct {
	// State is the game state BEFORE any moves in this "round" are applied.
	// It's only set at the start of each round (when SnakeIndex == 0).
	State *game.GameState

	// SnakeIndex is which snake is deciding at this node (0, 1, 2, ...)
	SnakeIndex int

	// SnakeID is the ID of the snake deciding at this node
	SnakeID string

	// Move is the move this snake chose to get here (set for all except root)
	Move int

	// MovesAccumulated are the moves chosen so far in this round
	// Key is snakeID, value is move
	MovesAccumulated map[string]int

	// Children are the possible moves for this snake
	Children []*AlternatingChild

	// Stats
	VisitCount int
	ValueSum   float32 // From THIS snake's perspective

	// Policy prior for this snake's moves
	Prior [4]float32

	// NNValues stores the neural network's value estimate for each snake.
	// This is populated during expansion and used for non-terminal evaluation.
	NNValues map[string]float32

	IsExpanded bool
	IsTerminal bool
}

// AlternatingChild represents a child edge (one move for the current snake)
type AlternatingChild struct {
	Move       int
	Node       *AlternatingNode
	PriorProb  float32
	VisitCount int
	ValueSum   float32 // From the parent snake's perspective
}

// AlternatingSearch runs MCTS where snakes take turns in the tree.
// This gives each snake its own UCB calculation from its own perspective.
func AlternatingSearch(ctx context.Context, client Predictor, config Config, rootState *game.GameState, simulations int) (*AlternatingNode, []string, error) {
	// Get consistent snake order (alive snakes only)
	snakeOrder := make([]string, 0, len(rootState.Snakes))
	for _, s := range rootState.Snakes {
		if s.Health > 0 {
			snakeOrder = append(snakeOrder, s.Id)
		}
	}

	if len(snakeOrder) == 0 {
		return nil, snakeOrder, nil
	}

	root := &AlternatingNode{
		State:            rootState.Clone(),
		SnakeIndex:       0,
		SnakeID:          snakeOrder[0],
		MovesAccumulated: make(map[string]int),
	}

	for i := 0; i < simulations; i++ {
		if ctx != nil {
			select {
			case <-ctx.Done():
				return root, snakeOrder, ctx.Err()
			default:
			}
		}

		// Selection & Expansion
		node, path, childPath := selectAndExpand(ctx, client, config, root, snakeOrder)
		if ctx != nil && ctx.Err() != nil {
			return root, snakeOrder, ctx.Err()
		}

		// Evaluation
		value := evaluate(node, snakeOrder)

		// Backpropagation - each node gets value from ITS snake's perspective
		backpropagate(path, childPath, snakeOrder, value)
	}

	return root, snakeOrder, nil
}

func selectAndExpand(ctx context.Context, client Predictor, config Config, root *AlternatingNode, snakeOrder []string) (*AlternatingNode, []*AlternatingNode, []*AlternatingChild) {
	node := root
	path := []*AlternatingNode{node}
	childPath := []*AlternatingChild{}

	for {
		if node.IsTerminal {
			break
		}

		if !node.IsExpanded {
			// Expand this node
			expandAlternatingNode(ctx, client, node, snakeOrder)
			break
		}

		if len(node.Children) == 0 {
			break
		}

		// Select best child using UCB from this snake's perspective
		bestChild := selectBestChild(node, config)
		if bestChild == nil {
			break
		}

		childPath = append(childPath, bestChild)
		node = bestChild.Node
		path = append(path, node)
	}

	return node, path, childPath
}

func selectBestChild(node *AlternatingNode, config Config) *AlternatingChild {
	var bestChild *AlternatingChild
	bestScore := float32(-1e9)
	sqrtN := float32(math.Sqrt(float64(node.VisitCount)))

	for _, child := range node.Children {
		q := float32(0)
		if child.VisitCount > 0 {
			q = child.ValueSum / float32(child.VisitCount)
		}

		u := q + config.Cpuct*child.PriorProb*sqrtN/(1+float32(child.VisitCount))

		if u > bestScore {
			bestScore = u
			bestChild = child
		}
	}

	return bestChild
}

func expandAlternatingNode(ctx context.Context, client Predictor, node *AlternatingNode, snakeOrder []string) {
	node.IsExpanded = true

	// Determine the state to use for this node
	var state *game.GameState
	if node.State != nil {
		state = node.State
	} else {
		// This shouldn't happen - state should be set when creating the node
		return
	}

	// Check if game is over
	if rules.IsGameOver(state) {
		node.IsTerminal = true
		return
	}

	// Get policy and value for ALL snakes at this state.
	// We need values for all snakes to properly evaluate non-terminal positions.
	node.NNValues = make(map[string]float32)
	var decidingSnakeState *game.GameState
	var decidingSnakePolicy [4]float32
	for _, snakeID := range snakeOrder {
		stateForSnake := state.Clone()
		stateForSnake.YouId = snakeID

		policyLogits, valueLogits, err := client.Predict(stateForSnake)
		if err != nil {
			continue
		}

		// Store value for this snake
		if len(valueLogits) > 0 {
			node.NNValues[snakeID] = valueLogits[0]
		}

		// Save policy for the current deciding snake
		if snakeID == node.SnakeID {
			decidingSnakePolicy = softmax4(policyLogits)
			node.Prior = decidingSnakePolicy
			decidingSnakeState = stateForSnake
		}
	}

	// Get legal moves for this snake (with tail decrement)
	if decidingSnakeState == nil {
		decidingSnakeState = state.Clone()
		decidingSnakeState.YouId = node.SnakeID
	}
	legalMoves := rules.GetLegalMovesWithTailDecrement(decidingSnakeState)
	if len(legalMoves) == 0 {
		legalMoves = []int{0} // Default if no legal moves
	}

	// Create children for each legal move
	node.Children = make([]*AlternatingChild, 0, len(legalMoves))

	for _, move := range legalMoves {
		// Accumulate this move
		newMoves := make(map[string]int, len(node.MovesAccumulated)+1)
		for k, v := range node.MovesAccumulated {
			newMoves[k] = v
		}
		newMoves[node.SnakeID] = move

		// Create child node
		child := &AlternatingChild{
			Move:      move,
			PriorProb: decidingSnakePolicy[move],
		}

		// Determine next snake or advance state
		nextSnakeIdx := node.SnakeIndex + 1

		if nextSnakeIdx >= len(snakeOrder) {
			// All snakes have chosen - advance the game state
			nextState := rules.NextStateSimultaneous(state, newMoves)

			// Get alive snakes in new state
			aliveSnakes := make([]string, 0)
			for _, s := range nextState.Snakes {
				if s.Health > 0 {
					aliveSnakes = append(aliveSnakes, s.Id)
				}
			}

			if len(aliveSnakes) == 0 || rules.IsGameOver(nextState) {
				child.Node = &AlternatingNode{
					State:            nextState,
					SnakeIndex:       0,
					SnakeID:          "",
					Move:             move,
					MovesAccumulated: make(map[string]int),
					IsTerminal:       true,
				}
			} else {
				child.Node = &AlternatingNode{
					State:            nextState,
					SnakeIndex:       0,
					SnakeID:          aliveSnakes[0],
					Move:             move,
					MovesAccumulated: make(map[string]int),
				}
			}
		} else {
			// Next snake in the same round
			child.Node = &AlternatingNode{
				State:            state, // Same state - round not complete yet
				SnakeIndex:       nextSnakeIdx,
				SnakeID:          snakeOrder[nextSnakeIdx],
				Move:             move,
				MovesAccumulated: newMoves,
			}
		}

		node.Children = append(node.Children, child)
	}
}

func evaluate(node *AlternatingNode, snakeOrder []string) map[string]float32 {
	values := make(map[string]float32)

	if node.State == nil {
		return values
	}

	// For terminal nodes, use actual game result
	if node.IsTerminal {
		for _, snakeID := range snakeOrder {
			state := node.State.Clone()
			state.YouId = snakeID
			values[snakeID] = rules.GetResult(state)
		}
		return values
	}

	// For non-terminal nodes, use neural network's value estimate.
	// This is the key insight from AlphaZero: the NN provides a heuristic
	// for positions we haven't fully explored.
	if node.NNValues != nil {
		for snakeID, nnValue := range node.NNValues {
			values[snakeID] = nnValue
		}
		return values
	}

	// Fallback: if no NN values available, use game result (may be 0 for ongoing)
	for _, snakeID := range snakeOrder {
		state := node.State.Clone()
		state.YouId = snakeID
		values[snakeID] = rules.GetResult(state)
	}

	return values
}

func backpropagate(path []*AlternatingNode, childPath []*AlternatingChild, snakeOrder []string, values map[string]float32) {
	// Update each node with the value from ITS snake's perspective
	for i, node := range path {
		node.VisitCount++
		if v, ok := values[node.SnakeID]; ok {
			node.ValueSum += v
		}

		// Update the child that led to the next node
		if i < len(childPath) {
			child := childPath[i]
			child.VisitCount++
			// Child value is from the parent node's snake perspective
			if v, ok := values[node.SnakeID]; ok {
				child.ValueSum += v
			}
		}
	}
}

// GetAlternatingSearchResult extracts the final move distribution for each snake
func GetAlternatingSearchResult(root *AlternatingNode, snakeOrder []string) *AlternatingSearchResult {
	result := &AlternatingSearchResult{
		SnakeOrder:   snakeOrder,
		Policies:     make(map[string][4]float32),
		RootChildren: make([]AlternatingChildSummary, 0),
	}

	if root == nil || len(root.Children) == 0 {
		return result
	}

	// For the first snake, get the visit distribution
	firstSnakePolicy := [4]float32{}
	totalVisits := 0
	for _, child := range root.Children {
		totalVisits += child.VisitCount
	}

	for _, child := range root.Children {
		if totalVisits > 0 {
			firstSnakePolicy[child.Move] = float32(child.VisitCount) / float32(totalVisits)
		}
		result.RootChildren = append(result.RootChildren, AlternatingChildSummary{
			SnakeID:    root.SnakeID,
			Move:       child.Move,
			VisitCount: child.VisitCount,
			ValueSum:   child.ValueSum,
			Q:          safeQ(child.ValueSum, child.VisitCount),
			PriorProb:  child.PriorProb,
		})
	}
	result.Policies[root.SnakeID] = firstSnakePolicy

	// For other snakes, we need to look at their level in the tree
	// This is approximate - we average over all reachable nodes for each snake
	for snakeIdx := 1; snakeIdx < len(snakeOrder); snakeIdx++ {
		snakeID := snakeOrder[snakeIdx]
		policy := [4]float32{}
		moveCounts := [4]int{}

		// Traverse to find all nodes at this snake's level and aggregate
		collectPoliciesAtLevel(root, snakeIdx, snakeOrder, &policy, &moveCounts)

		total := 0
		for _, c := range moveCounts {
			total += c
		}
		if total > 0 {
			for i := 0; i < 4; i++ {
				policy[i] = float32(moveCounts[i]) / float32(total)
			}
		}
		result.Policies[snakeID] = policy
	}

	return result
}

func collectPoliciesAtLevel(node *AlternatingNode, targetLevel int, snakeOrder []string, policy *[4]float32, moveCounts *[4]int) {
	if node.SnakeIndex == targetLevel && node.SnakeID == snakeOrder[targetLevel] {
		// This is a node at the target level
		for _, child := range node.Children {
			moveCounts[child.Move] += child.VisitCount
		}
		return
	}

	// Recurse into children
	for _, child := range node.Children {
		if child.Node != nil {
			collectPoliciesAtLevel(child.Node, targetLevel, snakeOrder, policy, moveCounts)
		}
	}
}

func safeQ(valueSum float32, visitCount int) float32 {
	if visitCount == 0 {
		return 0
	}
	return valueSum / float32(visitCount)
}

// AlternatingSearchResult contains the extracted policies and summaries
type AlternatingSearchResult struct {
	SnakeOrder   []string
	Policies     map[string][4]float32
	RootChildren []AlternatingChildSummary
}

// AlternatingChildSummary is a compact representation of a child at the root level
type AlternatingChildSummary struct {
	SnakeID    string  `json:"snake_id"`
	Move       int     `json:"move"`
	VisitCount int     `json:"n"`
	ValueSum   float32 `json:"value_sum"`
	Q          float32 `json:"q"`
	PriorProb  float32 `json:"p"`
}
