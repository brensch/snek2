package mcts

import (
	"context"
	"math"
	"math/rand"

	"github.com/brensch/snek2/game"
	"github.com/brensch/snek2/rules"
)

func softmax4(logits []float32) [4]float32 {
	var out [4]float32
	if len(logits) < 4 {
		return out
	}
	maxV := logits[0]
	for i := 1; i < 4; i++ {
		if logits[i] > maxV {
			maxV = logits[i]
		}
	}
	sum := float32(0)
	for i := 0; i < 4; i++ {
		e := float32(math.Exp(float64(logits[i] - maxV)))
		out[i] = e
		sum += e
	}
	if sum > 0 {
		inv := 1 / sum
		for i := 0; i < 4; i++ {
			out[i] *= inv
		}
	}
	return out
}

// sampleMove samples a move index from a probability distribution
func sampleMove(rng *rand.Rand, probs [4]float32) int {
	r := rng.Float32()
	cumulative := float32(0)
	for i := 0; i < 4; i++ {
		cumulative += probs[i]
		if r < cumulative {
			return i
		}
	}
	return 3 // fallback to last move
}

// getEnemyMoves runs inference on each enemy snake and picks their most likely move
func (m *MCTS) getEnemyMoves(state *game.GameState, rng *rand.Rand) (map[string]int, error) {
	moves := make(map[string]int)
	egoId := state.YouId

	for i := range state.Snakes {
		s := &state.Snakes[i]
		if s.Id == egoId || s.Health <= 0 || len(s.Body) == 0 {
			continue
		}

		// Create a state from this enemy's perspective
		enemyState := state.Clone()
		enemyState.YouId = s.Id

		// Get legal moves for this enemy
		legalMoves := rules.GetLegalMoves(enemyState)
		if len(legalMoves) == 0 {
			// No legal moves, snake will die - pick any move
			moves[s.Id] = 0
			continue
		}

		// Run inference to get enemy's policy
		policyLogits, _, err := m.Client.Predict(enemyState)
		if err != nil {
			return nil, err
		}

		priors := softmax4(policyLogits)

		// Find the best legal move (argmax over legal moves)
		bestMove := legalMoves[0]
		bestProb := float32(-1)
		for _, mv := range legalMoves {
			if priors[mv] > bestProb {
				bestProb = priors[mv]
				bestMove = mv
			}
		}

		moves[s.Id] = bestMove
	}

	return moves, nil
}

const (
	VirtualLoss = float32(1.0)
	BatchSize   = 16
)

// Search runs the MCTS simulations
func (m *MCTS) Search(ctx context.Context, rootState *game.GameState, simulations int) (*Node, int, error) {
	root := NewNode(rootState, 1.0)
	maxDepth := 0

	for i := 0; i < simulations; i++ {
		if ctx != nil {
			select {
			case <-ctx.Done():
				return root, maxDepth, ctx.Err()
			default:
			}
		}

		node := root
		path := []*Node{node}

		// Selection
		for node.IsExpanded {
			bestMove := Move(-1)
			bestScore := float32(-1e9)

			// Calculate sqrt(sum(N)) for parent
			sqrtSumN := float32(math.Sqrt(float64(node.VisitCount)))

			for moveIdx := 0; moveIdx < 4; moveIdx++ {
				move := Move(moveIdx)
				child := node.Children[move]
				if child == nil {
					continue
				}

				q := float32(0)
				if child.VisitCount > 0 {
					q = child.ValueSum / float32(child.VisitCount)
				}

				// PUCT formula
				// U(s,a) = Q(s,a) + C_puct * P(s,a) * sqrt(sum(N)) / (1 + N)
				u := q + m.Config.Cpuct*child.PriorProb*sqrtSumN/(1+float32(child.VisitCount))

				if u > bestScore {
					bestScore = u
					bestMove = move
				}
			}

			node = node.Children[bestMove]
			path = append(path, node)
		}

		if d := len(path) - 1; d > maxDepth {
			maxDepth = d
		}

		// Expansion & Evaluation
		value := float32(0)

		if rules.IsTerminal(node.State) {
			value = rules.GetResult(node.State)
		} else {
			if ctx != nil {
				select {
				case <-ctx.Done():
					return root, maxDepth, ctx.Err()
				default:
				}
			}

			// Inference for ego snake
			policyLogits, values, err := m.Client.Predict(node.State)
			if err != nil {
				return nil, 0, err
			}

			if len(values) > 0 {
				value = values[0]
			}

			priors := softmax4(policyLogits)

			// Expand: for each legal ego move, sample enemy moves and simulate simultaneously
			legalMoves := rules.GetLegalMoves(node.State)

			for _, moveInt := range legalMoves {
				if ctx != nil {
					select {
					case <-ctx.Done():
						return root, maxDepth, ctx.Err()
					default:
					}
				}

				move := Move(moveInt)
				prob := priors[int(move)]

				// Sample enemy moves independently for each ego move
				// This ensures we get different enemy responses for different ego actions
				enemyMoves, err := m.getEnemyMoves(node.State, m.Rng)
				if err != nil {
					return nil, 0, err
				}

				// Build moves map: ego move + sampled enemy moves
				allMoves := make(map[string]int)
				allMoves[node.State.YouId] = int(move)
				for id, mv := range enemyMoves {
					allMoves[id] = mv
				}

				// Simulate with all snakes moving simultaneously
				nextState := rules.NextStateSimultaneous(node.State, allMoves)
				child := NewNode(nextState, prob)
				node.Children[move] = child
			}
			node.IsExpanded = true
		}

		// Backpropagation
		for _, n := range path {
			n.VisitCount++
			n.ValueSum += value
		}
	}

	return root, maxDepth, nil
}
