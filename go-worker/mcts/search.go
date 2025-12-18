package mcts

import (
	"context"
	"math"
	"sync"

	pb "github.com/brensch/snek2/gen/go"
	"github.com/brensch/snek2/go-worker/convert"
	"github.com/brensch/snek2/rules"
)

const (
	VirtualLoss = float32(1.0)
	BatchSize   = 256
)

// Search runs the MCTS simulations
func (m *MCTS) Search(rootState *pb.GameState, simulations int) (*Node, error) {
	root := NewNode(rootState, 1.0)

	for i := 0; i < simulations; i += BatchSize {
		currentBatchSize := BatchSize
		if i+BatchSize > simulations {
			currentBatchSize = simulations - i
		}

		var wg sync.WaitGroup
		leaves := make([]*Node, currentBatchSize)
		paths := make([][]*Node, currentBatchSize)

		// 1. Selection (Parallel)
		for j := 0; j < currentBatchSize; j++ {
			wg.Add(1)
			go func(idx int) {
				defer wg.Done()
				node := root
				path := []*Node{node}

				for {
					node.mu.Lock()
					if len(node.Children) == 0 {
						// Leaf
						node.VisitCount++
						node.ValueSum -= VirtualLoss
						node.mu.Unlock()
						break
					}

					bestMove := Move(-1)
					bestScore := float32(-1e9)
					sqrtSumN := float32(math.Sqrt(float64(node.VisitCount)))

					for move, child := range node.Children {
						q := float32(0)
						if child.VisitCount > 0 {
							q = child.ValueSum / float32(child.VisitCount)
						}
						u := q + m.Config.Cpuct*child.PriorProb*sqrtSumN/(1+float32(child.VisitCount))

						if u > bestScore {
							bestScore = u
							bestMove = move
						}
					}

					child := node.Children[bestMove]
					// Apply Virtual Loss
					child.VisitCount++
					child.ValueSum -= VirtualLoss
					
					node.mu.Unlock()
					
					node = child
					path = append(path, node)
				}
				leaves[idx] = node
				paths[idx] = path
			}(j)
		}
		wg.Wait()

		// 2. Inference (Batched)
		var inferenceIndices []int
		var inferenceData []byte
		
		for j, leaf := range leaves {
			if rules.IsTerminal(leaf.State) {
				continue
			}
			
			inferenceIndices = append(inferenceIndices, j)
			dataPtr := convert.StateToBytes(leaf.State)
			inferenceData = append(inferenceData, *dataPtr...)
			convert.PutBuffer(dataPtr)
		}

		var responses []*pb.InferenceResponse
		if len(inferenceIndices) > 0 {
			req := &pb.InferenceRequest{
				Data:  inferenceData,
				Shape: []int32{int32(convert.Channels), int32(convert.Height), int32(convert.Width)},
			}
			resp, err := m.Client.Predict(context.Background(), req)
			if err != nil {
				return nil, err
			}
			responses = resp.Responses
		}

		// 3. Expansion & Backprop
		for j := 0; j < currentBatchSize; j++ {
			leaf := leaves[j]
			path := paths[j]
			value := float32(0)
			
			// Check if this leaf was part of inference
			infIdx := -1
			for k, idx := range inferenceIndices {
				if idx == j {
					infIdx = k
					break
				}
			}

			if infIdx != -1 {
				// It was inferred
				infResp := responses[infIdx]
				value = infResp.Value
				
				leaf.mu.Lock()
				if len(leaf.Children) == 0 {
					// Expand
					legalMoves := rules.GetLegalMoves(leaf.State)
					for _, moveInt := range legalMoves {
						move := Move(moveInt)
						prob := float32(0)
						if int(move) < len(infResp.Policy) {
							prob = infResp.Policy[int(move)]
						}
						nextState := rules.NextState(leaf.State, int(move))
						leaf.Children[move] = NewNode(nextState, prob)
					}
				}
				leaf.mu.Unlock()
			} else {
				// Terminal
				value = rules.GetResult(leaf.State)
			}

			// Backprop
			// Undo Virtual Loss and add Real Value
			for _, n := range path {
				n.mu.Lock()
				// We added +1 visit, -1 value (VirtualLoss)
				// Now we want: +1 visit (already done), +value
				// So we add: value + VirtualLoss
				n.ValueSum += value + VirtualLoss
				n.mu.Unlock()
			}
		}
	}

	return root, nil
}
