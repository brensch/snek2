package mcts

import (
	"context"
	"math"

	pb "github.com/brensch/snek2/gen/go"
	"github.com/brensch/snek2/go-worker/convert"
	"github.com/brensch/snek2/rules"
)

// Search runs the MCTS simulations
func (m *MCTS) Search(rootState *pb.GameState, simulations int) (*Node, error) {
	root := NewNode(rootState, 1.0)

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
				return nil, err
			}

			// We assume the server returns one response for our one request
			if len(resp.Responses) > 0 {
				infResp := resp.Responses[0]
				value = infResp.Value

				// Expand
				legalMoves := rules.GetLegalMoves(node.State)
				for _, moveInt := range legalMoves {
					move := Move(moveInt)
					
					// Get probability from policy
					prob := float32(0)
					if int(move) < len(infResp.Policy) {
						prob = infResp.Policy[int(move)]
					}

					// Create child
					nextState := rules.NextState(node.State, int(move))
					child := NewNode(nextState, prob)
					node.Children[move] = child
				}
			}
		}

		// Backpropagation
		for _, n := range path {
			n.VisitCount++
			n.ValueSum += value
		}
	}

	return root, nil
}
