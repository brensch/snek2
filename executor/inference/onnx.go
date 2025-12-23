package inference

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/brensch/snek2/executor/convert"
	"github.com/brensch/snek2/game"
	ort "github.com/yalue/onnxruntime_go"
)

const (
	InputSize  = 14 * 11 * 11
	PolicySize = 4
	ValueSize  = 1
)

const (
	DefaultBatchSize    = 128
	DefaultBatchTimeout = 1 * time.Millisecond
)

type OnnxClientConfig struct {
	BatchSize    int
	BatchTimeout time.Duration
}

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
	cfg          OnnxClientConfig
}

var ortInitOnce sync.Once
var ortInitErr error

func NewOnnxClient(modelPath string) (*OnnxClient, error) {
	return NewOnnxClientWithConfig(modelPath, OnnxClientConfig{BatchSize: DefaultBatchSize, BatchTimeout: DefaultBatchTimeout})
}

func NewOnnxClientWithConfig(modelPath string, cfg OnnxClientConfig) (*OnnxClient, error) {
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = DefaultBatchSize
	}
	if cfg.BatchTimeout <= 0 {
		cfg.BatchTimeout = DefaultBatchTimeout
	}

	if runtime.GOOS == "linux" {
		ensureLinuxLibraryPath()
		if p := os.Getenv("ORT_SHARED_LIBRARY_PATH"); p != "" {
			ort.SetSharedLibraryPath(p)
		} else {
			cwd, _ := os.Getwd()
			candidates := []string{
				"libonnxruntime.so",
				"libonnxruntime.so.1",
				"libonnxruntime.so.1.23.2",
			}
			for _, name := range candidates {
				abs := filepath.Join(cwd, name)
				if _, err := os.Stat(abs); err == nil {
					ort.SetSharedLibraryPath(abs)
					break
				}
			}
		}
	}

	ortInitOnce.Do(func() {
		ortInitErr = ort.InitializeEnvironment()
	})
	if ortInitErr != nil {
		return nil, fmt.Errorf("failed to init ort: %w", ortInitErr)
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
		cfg:          cfg,
		requestsChan: make(chan inferenceRequest, cfg.BatchSize*2),
	}

	go client.batchLoop()

	return client, nil
}

func ensureLinuxLibraryPath() {
	cwd, err := os.Getwd()
	if err != nil {
		return
	}

	// These are the common locations of CUDA + Torch shared libraries when installed
	// via pip packages inside the project's .venv.
	// Don't hardcode python version; glob python*/site-packages.
	candidateDirs := []string{cwd}

	patterns := []string{
		filepath.Join(cwd, ".venv", "lib", "python*", "site-packages", "nvidia", "*", "lib"),
		filepath.Join(cwd, ".venv", "lib", "python*", "site-packages", "triton", "backends", "nvidia", "lib"),
		filepath.Join(cwd, ".venv", "lib", "python*", "site-packages", "torch", "lib"),
	}
	for _, pat := range patterns {
		matches, _ := filepath.Glob(pat)
		for _, m := range matches {
			candidateDirs = append(candidateDirs, m)
		}
	}

	existing := os.Getenv("LD_LIBRARY_PATH")
	existingSet := map[string]bool{}
	for _, p := range strings.Split(existing, ":") {
		if p == "" {
			continue
		}
		existingSet[p] = true
	}

	toAdd := make([]string, 0, len(candidateDirs))
	for _, d := range candidateDirs {
		if existingSet[d] {
			continue
		}
		if st, err := os.Stat(d); err == nil && st.IsDir() {
			toAdd = append(toAdd, d)
		}
	}
	if len(toAdd) == 0 {
		return
	}

	newVal := strings.Join(toAdd, ":")
	if existing != "" {
		newVal = newVal + ":" + existing
	}
	_ = os.Setenv("LD_LIBRARY_PATH", newVal)
}

func (c *OnnxClient) Close() error {
	return c.session.Destroy()
}

func (c *OnnxClient) Predict(state *game.GameState) ([]float32, []float32, error) {
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
	batchInput := make([]float32, 0, c.cfg.BatchSize*InputSize)
	requests := make([]inferenceRequest, 0, c.cfg.BatchSize)

	ticker := time.NewTicker(c.cfg.BatchTimeout)
	defer ticker.Stop()

	for {
		select {
		case req := <-c.requestsChan:
			requests = append(requests, req)
			batchInput = append(batchInput, req.input...)

			if len(requests) >= c.cfg.BatchSize {
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
	inputShape := []int64{currentBatchSize, 14, 11, 11}
	inputTensor, err := ort.NewTensor(ort.NewShape(inputShape...), batchInput)
	if err != nil {
		c.failBatch(requests, err)
		return
	}
	defer inputTensor.Destroy()

	// Create output tensors
	policyShape := []int64{currentBatchSize, 4}
	policyTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(policyShape...))
	if err != nil {
		c.failBatch(requests, err)
		return
	}
	defer policyTensor.Destroy()

	valueShape := []int64{currentBatchSize, 1}
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
