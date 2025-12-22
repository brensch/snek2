package inference

import (
	"fmt"
	"math"
	"runtime"
	"time"

	"github.com/brensch/snek2/executor/convert"
	pb "github.com/brensch/snek2/gen/go"
	ort "github.com/yalue/onnxruntime_go"
)

const (
	BatchSize    = 128
	BatchTimeout = 1 * time.Millisecond
	InputSize    = 17 * 11 * 11
	PolicySize   = 16
	ValueSize    = 4
)

type inferenceRequest struct {
	input    []float32
	respChan chan inferenceResponse
}

type inferenceResponse struct {
	policy []float32
	value  []float32
	err    error
}

// OnnxClient implements the inference engine using ONNX Runtime with batching
type OnnxClient struct {
	session      *ort.DynamicAdvancedSession
	requestsChan chan inferenceRequest
}

func NewOnnxClient(modelPath string) (*OnnxClient, error) {
	if runtime.GOOS == "linux" {
		ort.SetSharedLibraryPath("libonnxruntime.so")
	}

	err := ort.InitializeEnvironment()
	if err != nil {
		return nil, fmt.Errorf("failed to init ort: %w", err)
	}

	options, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}
	defer options.Destroy()

	// Create session
	inputs := []string{"input"}
	outputs := []string{"policy", "value"}

	// Set intra-op threads to 1 to avoid contention since we have many workers
	options.SetIntraOpNumThreads(1)
	options.SetInterOpNumThreads(1)

	// Try to use CUDA if available
	cudaOptions, err := ort.NewCUDAProviderOptions()
	if err == nil {
		defer cudaOptions.Destroy()
		err = options.AppendExecutionProviderCUDA(cudaOptions)
		if err != nil {
			fmt.Println("Failed to append CUDA provider:", err)
		} else {
			fmt.Println("CUDA provider enabled!")
		}
	} else {
		fmt.Println("Failed to create CUDA options:", err)
	}

	session, err := ort.NewDynamicAdvancedSession(modelPath, inputs, outputs, options)
	if err != nil {
		return nil, fmt.Errorf("failed to create session: %w", err)
	}

	client := &OnnxClient{
		session:      session,
		requestsChan: make(chan inferenceRequest, BatchSize*2),
	}

	go client.batchLoop()

	return client, nil
}

func (c *OnnxClient) Close() error {
	return c.session.Destroy()
}

func (c *OnnxClient) Predict(state *pb.GameState) ([]float32, []float32, error) {
	// Convert state to tensor data
	byteDataPtr := convert.StateToBytes(state)
	byteData := *byteDataPtr

	// Convert bytes to float32 slice
	floats := make([]float32, InputSize)
	for i := 0; i < InputSize; i++ {
		bits := uint32(byteData[i*4]) | uint32(byteData[i*4+1])<<8 | uint32(byteData[i*4+2])<<16 | uint32(byteData[i*4+3])<<24
		floats[i] = math.Float32frombits(bits)
	}
	convert.PutBuffer(byteDataPtr)

	respChan := make(chan inferenceResponse, 1)
	c.requestsChan <- inferenceRequest{
		input:    floats,
		respChan: respChan,
	}

	resp := <-respChan
	return resp.policy, resp.value, resp.err
}

func (c *OnnxClient) batchLoop() {
	batchInput := make([]float32, 0, BatchSize*InputSize)
	requests := make([]inferenceRequest, 0, BatchSize)

	ticker := time.NewTicker(BatchTimeout)
	defer ticker.Stop()

	for {
		select {
		case req := <-c.requestsChan:
			requests = append(requests, req)
			batchInput = append(batchInput, req.input...)

			if len(requests) >= BatchSize {
				c.runBatch(requests, batchInput)
				// Reset
				requests = requests[:0]
				batchInput = batchInput[:0]
			}
		case <-ticker.C:
			if len(requests) > 0 {
				c.runBatch(requests, batchInput)
				// Reset
				requests = requests[:0]
				batchInput = batchInput[:0]
			}
		}
	}
}

func (c *OnnxClient) runBatch(requests []inferenceRequest, batchInput []float32) {
	currentBatchSize := int64(len(requests))

	// Create input tensor
	inputShape := []int64{currentBatchSize, 17, 11, 11}
	inputTensor, err := ort.NewTensor(ort.NewShape(inputShape...), batchInput)
	if err != nil {
		c.failBatch(requests, err)
		return
	}
	defer inputTensor.Destroy()

	// Create output tensors
	policyShape := []int64{currentBatchSize, 16}
	policyTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(policyShape...))
	if err != nil {
		c.failBatch(requests, err)
		return
	}
	defer policyTensor.Destroy()

	valueShape := []int64{currentBatchSize, 4}
	valueTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(valueShape...))
	if err != nil {
		c.failBatch(requests, err)
		return
	}
	defer valueTensor.Destroy()

	// Run inference
	err = c.session.Run([]ort.Value{inputTensor}, []ort.Value{policyTensor, valueTensor})
	if err != nil {
		c.failBatch(requests, err)
		return
	}

	// Get outputs
	policyData := policyTensor.GetData()
	valueData := valueTensor.GetData()

	// Distribute results
	for i, req := range requests {
		// Copy policy
		policy := make([]float32, PolicySize)
		copy(policy, policyData[i*PolicySize:(i+1)*PolicySize])

		// Copy value
		value := make([]float32, ValueSize)
		copy(value, valueData[i*ValueSize:(i+1)*ValueSize])

		req.respChan <- inferenceResponse{
			policy: policy,
			value:  value,
			err:    nil,
		}
	}
}

func (c *OnnxClient) failBatch(requests []inferenceRequest, err error) {
	for _, req := range requests {
		req.respChan <- inferenceResponse{err: err}
	}
}
