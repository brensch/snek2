package mcts

import (
	"context"
	"math"
	"sort"

	pb "github.com/brensch/snek2/gen/go"
	"github.com/brensch/snek2/go-worker/convert"
	"github.com/brensch/snek2/rules"
)

const (
	VirtualLoss = float32(1.0)
	BatchSize   = 16
)

// Search runs the MCTS simulations
func (m *MCTS) Search(rootState *pb.GameState, simulations int) (*Node, int, error) {
	root := NewNode(rootState, 1.0)
	maxDepth := 0

	for i := 0; i < simulations; i++ {
		node := root
		path := []*Node{node}

		// Selection
		for len(node.Children) > 0 {
			bestMove := Move(-1)
			bestScore := float32(-1e9)

			// Calculate sqrt(sum(N)) for parent
			sqrtSumN := float32(math.Sqrt(float64(node.VisitCount)))

			for move, child := range node.Children {
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
			// Inference
			dataPtr := convert.StateToBytes(node.State)

			req := &pb.InferenceRequest{
				Data:  *dataPtr,
				Shape: []int32{int32(convert.Channels), int32(convert.Height), int32(convert.Width)},
			}

			resp, err := m.Client.Predict(context.Background(), req)
			convert.PutBuffer(dataPtr) // Return buffer to pool
			if err != nil {
				return nil, 0, err
			}

			// We assume the server returns one response for our one request
			if len(resp.Responses) > 0 {
				infResp := resp.Responses[0]

				// Find index of YouId in sorted snakes to match model output
				ids := make([]string, len(node.State.Snakes))
				for i, s := range node.State.Snakes {
					ids[i] = s.Id
				}
				sort.Strings(ids)

				myIndex := -1
				for i, id := range ids {
					if id == node.State.YouId {
						myIndex = i
						break
					}
				}

				if myIndex != -1 {
					if myIndex < len(infResp.Values) {
						value = infResp.Values[myIndex]
					}

					// Expand
					legalMoves := rules.GetLegalMoves(node.State)
					for _, moveInt := range legalMoves {
						move := Move(moveInt)

						// Get probability from policy
						prob := float32(0)
						// Policy is flat array: [Snake0_Move0..3, Snake1_Move0..3, ...]
						policyIdx := (myIndex * 4) + int(move)
						if policyIdx < len(infResp.Policies) {
							prob = infResp.Policies[policyIdx]
						}

						// Create child
						nextState := rules.NextState(node.State, int(move))
						child := NewNode(nextState, prob)
						node.Children[move] = child
					}
				}
			}
		}

		// Backpropagation
		for _, n := range path {
			n.VisitCount++
			n.ValueSum += value
		}
	}

	return root, maxDepth, nil
}
