package mcts

import (
	"context"
	"math"

	"github.com/brensch/snek2/game"
	"github.com/brensch/snek2/rules"
)

// SimultaneousNode represents a node in the MCTS tree with proper simultaneous semantics.
// The tree alternates between snakes for exploration efficiency, but all snakes in a
// "round" see the same game state and use cached inference from that state.
// Moves are only applied when all snakes have chosen.
type SimultaneousNode struct {
	// State is the game state at the START of this round.
	// All snakes in this round see this same state.
	State *game.GameState

	// SnakeIndex is which snake is deciding at this node (0, 1, 2, ...)
	SnakeIndex int

	// SnakeID is the ID of the snake deciding at this node
	SnakeID string

	// Move is the move this snake chose to get to this node
	Move int

	// MovesAccumulated are the moves chosen so far in this round
	MovesAccumulated map[string]int

	// Children are the possible moves for this snake
	Children []*SimultaneousChild

	// Stats
	VisitCount int
	ValueSum   float32 // From THIS snake's perspective

	// RoundCache stores inference results computed once at the start of each round.
	// All nodes in the same round share this cache (via pointer).
	RoundCache *RoundInferenceCache

	IsExpanded bool
	IsTerminal bool
}

// RoundInferenceCache holds cached NN predictions for all snakes at a given state.
// This is computed once when the round starts (SnakeIndex == 0) and shared
// by all nodes in that round.
type RoundInferenceCache struct {
	// Policies for each snake, computed on the round's base state
	Policies map[string][4]float32

	// Values for each snake, computed on the round's base state
	Values map[string]float32

	// LegalMoves for each snake, computed on the round's base state
	LegalMoves map[string][]int

	// AggregatedStats tracks move statistics for each snake, aggregated across
	// all branches of earlier snakes. This is what makes the search "simultaneous" -
	// later snakes' UCB selection uses these aggregated stats, not branch-specific ones.
	// Key: snakeID, Value: [4]MoveStats for each move direction
	AggregatedStats map[string]*[4]MoveStats
}

// MoveStats tracks aggregated statistics for a single move across all branches.
type MoveStats struct {
	VisitCount int
	ValueSum   float32
}

// SimultaneousChild represents a child edge (one move for the current snake)
type SimultaneousChild struct {
	Move       int
	Node       *SimultaneousNode
	PriorProb  float32
	VisitCount int
	ValueSum   float32 // From the parent snake's perspective
}

// SimultaneousSearch runs MCTS with alternating tree structure but simultaneous semantics.
// Key properties:
// 1. All snakes in a round see the same game state
// 2. NN inference is cached once per round for all snakes (efficiency)
// 3. Moves are applied simultaneously at the end of each round
// 4. No information leaks between snakes within a round
func SimultaneousSearch(ctx context.Context, client Predictor, config Config, rootState *game.GameState, simulations int) (*SimultaneousNode, []string, error) {
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

	root := &SimultaneousNode{
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
		node, path, childPath := simSelectAndExpand(ctx, client, config, root, snakeOrder)
		if ctx != nil && ctx.Err() != nil {
			return root, snakeOrder, ctx.Err()
		}

		// Evaluation
		value := simEvaluate(node, snakeOrder)

		// Backpropagation - each node gets value from ITS snake's perspective
		simBackpropagate(path, childPath, snakeOrder, value)
	}

	return root, snakeOrder, nil
}

func simSelectAndExpand(ctx context.Context, client Predictor, config Config, root *SimultaneousNode, snakeOrder []string) (*SimultaneousNode, []*SimultaneousNode, []*SimultaneousChild) {
	node := root
	path := []*SimultaneousNode{node}
	childPath := []*SimultaneousChild{}

	for {
		if node.IsTerminal {
			break
		}

		if !node.IsExpanded {
			simExpandNode(ctx, client, node, snakeOrder)
			break
		}

		if len(node.Children) == 0 {
			break
		}

		// Select best child using UCB from this snake's perspective
		bestChild := simSelectBestChild(node, config)
		if bestChild == nil {
			break
		}

		childPath = append(childPath, bestChild)
		node = bestChild.Node
		path = append(path, node)
	}

	return node, path, childPath
}

func simSelectBestChild(node *SimultaneousNode, config Config) *SimultaneousChild {
	var bestChild *SimultaneousChild
	bestScore := float32(-1e9)
	sqrtN := float32(math.Sqrt(float64(node.VisitCount)))

	// For snakes after the first in a round (SnakeIndex > 0), use aggregated
	// statistics instead of branch-specific ones. This ensures the snake's
	// exploration is NOT conditioned on earlier snakes' choices.
	useAggregated := node.SnakeIndex > 0 && node.RoundCache != nil &&
		node.RoundCache.AggregatedStats != nil

	var aggStats *[4]MoveStats
	var aggTotalVisits int
	if useAggregated {
		aggStats = node.RoundCache.AggregatedStats[node.SnakeID]
		if aggStats != nil {
			for i := 0; i < 4; i++ {
				aggTotalVisits += aggStats[i].VisitCount
			}
		}
	}

	sqrtAggN := float32(1)
	if aggTotalVisits > 0 {
		sqrtAggN = float32(math.Sqrt(float64(aggTotalVisits)))
	}

	for _, child := range node.Children {
		var q float32
		var n int

		if useAggregated && aggStats != nil {
			// Use aggregated statistics across all branches
			stats := &aggStats[child.Move]
			n = stats.VisitCount
			if n > 0 {
				q = stats.ValueSum / float32(n)
			}
			// UCB with aggregated stats
			u := q + config.Cpuct*child.PriorProb*sqrtAggN/(1+float32(n))
			if u > bestScore {
				bestScore = u
				bestChild = child
			}
		} else {
			// First snake in round: use branch-specific statistics (normal MCTS)
			n = child.VisitCount
			if n > 0 {
				q = child.ValueSum / float32(n)
			}
			u := q + config.Cpuct*child.PriorProb*sqrtN/(1+float32(n))
			if u > bestScore {
				bestScore = u
				bestChild = child
			}
		}
	}

	return bestChild
}

// simExpandNode expands a node using cached inference from the round's base state.
func simExpandNode(ctx context.Context, client Predictor, node *SimultaneousNode, snakeOrder []string) {
	node.IsExpanded = true

	state := node.State
	if state == nil {
		return
	}

	// Check if game is over
	if rules.IsGameOver(state) {
		node.IsTerminal = true
		return
	}

	// Get or create the round cache.
	// If this is the first snake in the round (SnakeIndex == 0), compute fresh cache.
	// Otherwise, the cache should have been passed from the parent.
	if node.RoundCache == nil {
		node.RoundCache = computeRoundCache(ctx, client, state, snakeOrder)
	}

	cache := node.RoundCache
	if cache == nil {
		return
	}

	// Get this snake's policy and legal moves from the cache
	policy := cache.Policies[node.SnakeID]
	legalMoves := cache.LegalMoves[node.SnakeID]

	if len(legalMoves) == 0 {
		legalMoves = []int{0} // Default if no legal moves
	}

	// Create children for each legal move
	node.Children = make([]*SimultaneousChild, 0, len(legalMoves))

	for _, move := range legalMoves {
		// Accumulate this move
		newMoves := make(map[string]int, len(node.MovesAccumulated)+1)
		for k, v := range node.MovesAccumulated {
			newMoves[k] = v
		}
		newMoves[node.SnakeID] = move

		child := &SimultaneousChild{
			Move:      move,
			PriorProb: policy[move],
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
				child.Node = &SimultaneousNode{
					State:            nextState,
					SnakeIndex:       0,
					SnakeID:          "",
					Move:             move,
					MovesAccumulated: make(map[string]int),
					IsTerminal:       true,
					// New round, new cache will be computed when expanded
					RoundCache: nil,
				}
			} else {
				child.Node = &SimultaneousNode{
					State:            nextState,
					SnakeIndex:       0,
					SnakeID:          aliveSnakes[0],
					Move:             move,
					MovesAccumulated: make(map[string]int),
					// New round, new cache will be computed when expanded
					RoundCache: nil,
				}
			}
		} else {
			// Next snake in the same round - SHARE the same cache and state
			child.Node = &SimultaneousNode{
				State:            state, // Same state - round not complete yet
				SnakeIndex:       nextSnakeIdx,
				SnakeID:          snakeOrder[nextSnakeIdx],
				Move:             move,
				MovesAccumulated: newMoves,
				// Share the round cache - this is the key difference!
				RoundCache: cache,
			}
		}

		node.Children = append(node.Children, child)
	}
}

// computeRoundCache runs inference for all snakes on the given state.
// This is called once at the start of each round.
func computeRoundCache(ctx context.Context, client Predictor, state *game.GameState, snakeOrder []string) *RoundInferenceCache {
	cache := &RoundInferenceCache{
		Policies:        make(map[string][4]float32),
		Values:          make(map[string]float32),
		LegalMoves:      make(map[string][]int),
		AggregatedStats: make(map[string]*[4]MoveStats),
	}

	// Get alive snakes
	aliveSnakes := make([]string, 0)
	for _, s := range state.Snakes {
		if s.Health > 0 {
			aliveSnakes = append(aliveSnakes, s.Id)
		}
	}

	for _, snakeID := range aliveSnakes {
		if ctx != nil {
			select {
			case <-ctx.Done():
				return cache
			default:
			}
		}

		stateForSnake := state.Clone()
		stateForSnake.YouId = snakeID

		// Get NN predictions
		policyLogits, valueLogits, err := client.Predict(stateForSnake)
		if err != nil {
			continue
		}

		cache.Policies[snakeID] = softmax4(policyLogits)
		if len(valueLogits) > 0 {
			cache.Values[snakeID] = valueLogits[0]
		}

		// Get legal moves (computed on the base state, not affected by other snakes' choices)
		cache.LegalMoves[snakeID] = rules.GetLegalMovesWithTailDecrement(stateForSnake)
		if len(cache.LegalMoves[snakeID]) == 0 {
			cache.LegalMoves[snakeID] = []int{0}
		}

		// Initialize aggregated stats for this snake (all zeros)
		cache.AggregatedStats[snakeID] = &[4]MoveStats{}
	}

	return cache
}

func simEvaluate(node *SimultaneousNode, snakeOrder []string) map[string]float32 {
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

	// For non-terminal nodes, use the cached NN values from the round
	if node.RoundCache != nil && len(node.RoundCache.Values) > 0 {
		for snakeID, val := range node.RoundCache.Values {
			values[snakeID] = val
		}
		return values
	}

	// Fallback: if no cached values, return zeros
	for _, snakeID := range snakeOrder {
		values[snakeID] = 0
	}

	return values
}

func simBackpropagate(path []*SimultaneousNode, childPath []*SimultaneousChild, snakeOrder []string, values map[string]float32) {
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

			// CRITICAL: Update aggregated stats in the round cache.
			// This allows later snakes' UCB selection to use statistics that are
			// NOT conditioned on earlier snakes' choices in this round.
			if node.RoundCache != nil && node.RoundCache.AggregatedStats != nil {
				if aggStats := node.RoundCache.AggregatedStats[node.SnakeID]; aggStats != nil {
					aggStats[child.Move].VisitCount++
					if v, ok := values[node.SnakeID]; ok {
						aggStats[child.Move].ValueSum += v
					}
				}
			}
		}
	}
}

// GetSimultaneousSearchResult extracts the final move distribution for each snake.
// This is what gets used for training - policies are based on visit counts.
func GetSimultaneousSearchResult(root *SimultaneousNode, snakeOrder []string) *SimultaneousSearchResult {
	result := &SimultaneousSearchResult{
		SnakeOrder:   snakeOrder,
		Policies:     make(map[string][4]float32),
		Values:       make(map[string]float32),
		RootChildren: make([]SimultaneousChildSummary, 0),
	}

	if root == nil || len(root.Children) == 0 {
		return result
	}

	// For the first snake, get the visit distribution directly
	firstSnakePolicy := [4]float32{}
	totalVisits := 0
	for _, child := range root.Children {
		totalVisits += child.VisitCount
	}

	for _, child := range root.Children {
		if totalVisits > 0 {
			firstSnakePolicy[child.Move] = float32(child.VisitCount) / float32(totalVisits)
		}
		result.RootChildren = append(result.RootChildren, SimultaneousChildSummary{
			SnakeID:    root.SnakeID,
			Move:       child.Move,
			VisitCount: child.VisitCount,
			ValueSum:   child.ValueSum,
			Q:          safeQ(child.ValueSum, child.VisitCount),
			PriorProb:  child.PriorProb,
		})
	}
	result.Policies[root.SnakeID] = firstSnakePolicy

	// For other snakes, use the aggregated stats from the round cache.
	// This is important: we DON'T want to condition snake B's policy on snake A's choice.
	// The aggregated stats track each snake's move statistics across ALL branches.
	if root.RoundCache != nil && root.RoundCache.AggregatedStats != nil {
		for snakeIdx := 1; snakeIdx < len(snakeOrder); snakeIdx++ {
			snakeID := snakeOrder[snakeIdx]
			aggStats := root.RoundCache.AggregatedStats[snakeID]
			if aggStats == nil {
				continue
			}

			total := 0
			for i := 0; i < 4; i++ {
				total += aggStats[i].VisitCount
			}
			if total > 0 {
				var policy [4]float32
				for i := 0; i < 4; i++ {
					policy[i] = float32(aggStats[i].VisitCount) / float32(total)
				}
				result.Policies[snakeID] = policy
			}
		}
	}

	// Get values from the round cache if available
	if root.RoundCache != nil {
		for snakeID, val := range root.RoundCache.Values {
			result.Values[snakeID] = val
		}
	}

	return result
}

func collectMoveCountsAtLevel(node *SimultaneousNode, targetLevel int, snakeOrder []string, moveCounts *[4]int) {
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
			collectMoveCountsAtLevel(child.Node, targetLevel, snakeOrder, moveCounts)
		}
	}
}

// SimultaneousSearchResult contains the extracted policies and summaries
type SimultaneousSearchResult struct {
	SnakeOrder   []string
	Policies     map[string][4]float32
	Values       map[string]float32
	RootChildren []SimultaneousChildSummary
}

// SimultaneousChildSummary is a compact representation of a child at the root level
type SimultaneousChildSummary struct {
	SnakeID    string  `json:"snake_id"`
	Move       int     `json:"move"`
	VisitCount int     `json:"n"`
	ValueSum   float32 `json:"value_sum"`
	Q          float32 `json:"q"`
	PriorProb  float32 `json:"p"`
}
