package inference

import (
	"fmt"
	"sync/atomic"

	"github.com/brensch/snek2/game"
)

// OnnxPool fans out Predict calls across multiple OnnxClient instances.
// Each client has its own batching loop + ORT session, allowing parallel
// inference execution on the GPU.
//
// Note: ORT environment initialization is process-global; OnnxClient handles
// that internally.
type OnnxPool struct {
	clients []*OnnxClient
	rr      atomic.Uint64
}

func (p *OnnxPool) Stats() RuntimeStats {
	var batches int64
	var items int64
	var runNanos int64
	var last int64
	queue := 0

	for _, c := range p.clients {
		st := c.Stats()
		batches += st.TotalBatches
		items += st.TotalItems
		runNanos += st.TotalRunNanos
		queue += st.QueueLen
		if st.LastBatchSize > last {
			last = st.LastBatchSize
		}
	}

	avgBatch := 0.0
	avgRunMs := 0.0
	if batches > 0 {
		avgBatch = float64(items) / float64(batches)
		avgRunMs = (float64(runNanos) / 1e6) / float64(batches)
	}

	return RuntimeStats{
		TotalBatches:  batches,
		TotalItems:    items,
		TotalRunNanos: runNanos,
		LastBatchSize: last,
		QueueLen:      queue,
		AvgBatchSize:  avgBatch,
		AvgRunMs:      avgRunMs,
	}
}

func NewOnnxClientPool(modelPath string, sessions int) (*OnnxPool, error) {
	return NewOnnxClientPoolWithConfig(modelPath, sessions, OnnxClientConfig{BatchSize: DefaultBatchSize, BatchTimeout: DefaultBatchTimeout})
}

func NewOnnxClientPoolWithConfig(modelPath string, sessions int, cfg OnnxClientConfig) (*OnnxPool, error) {
	if sessions <= 0 {
		sessions = 1
	}

	clients := make([]*OnnxClient, 0, sessions)
	for i := 0; i < sessions; i++ {
		c, err := NewOnnxClientWithConfig(modelPath, cfg)
		if err != nil {
			for _, created := range clients {
				_ = created.Close()
			}
			return nil, fmt.Errorf("create onnx client %d/%d: %w", i+1, sessions, err)
		}
		clients = append(clients, c)
	}

	return &OnnxPool{clients: clients}, nil
}

func (p *OnnxPool) Close() error {
	var firstErr error
	for _, c := range p.clients {
		if err := c.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

func (p *OnnxPool) Predict(state *game.GameState) ([]float32, []float32, error) {
	if len(p.clients) == 0 {
		return nil, nil, fmt.Errorf("onnx pool has no clients")
	}
	idx := int(p.rr.Add(1)-1) % len(p.clients)
	return p.clients[idx].Predict(state)
}
