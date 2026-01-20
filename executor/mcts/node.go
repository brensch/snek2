// Package mcts provides Monte Carlo Tree Search implementations for Battlesnake.
//
// The package includes two search algorithms:
//   - SimultaneousSearch: Tree alternates between snakes, but all see same state per round.
//     Moves are applied simultaneously at round end. Used for selfplay and viewer MCTS explorer.
//   - UnifiedSearch: All snakes explore jointly with combined action space.
//     Used for debug game generation with full tree visualization.
//
// Both algorithms use neural network predictions (via Predictor interface) for policy
// priors and value estimates.
package mcts

import (
	"math"
	"math/rand"

	"github.com/brensch/snek2/game"
)

// Move represents a direction (0: Up, 1: Down, 2: Left, 3: Right)
type Move int

// Config holds MCTS configuration
type Config struct {
	Cpuct float32 // Exploration constant for UCB formula
}

// Predictor defines the interface for neural network inference.
// Predict returns (policy_logits, values, error) for the given game state.
type Predictor interface {
	Predict(state *game.GameState) ([]float32, []float32, error)
}

// softmax4 converts 4 logits to a normalized probability distribution.
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

// safeQ computes Q = ValueSum / VisitCount, returning 0 if VisitCount is 0.
func safeQ(valueSum float32, visitCount int) float32 {
	if visitCount == 0 {
		return 0
	}
	return valueSum / float32(visitCount)
}
