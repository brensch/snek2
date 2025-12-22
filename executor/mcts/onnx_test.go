package mcts

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"testing"

	"github.com/brensch/snek2/executor/convert"
	"github.com/brensch/snek2/game"
	ort "github.com/yalue/onnxruntime_go"
)

// OnnxInferenceClient implements the InferenceServiceClient interface but runs locally via ONNX Runtime
type OnnxInferenceClient struct {
	session      *ort.DynamicAdvancedSession
	input        *ort.Tensor[float32]
	outputPolicy *ort.Tensor[float32]
	outputValue  *ort.Tensor[float32]
}

func NewOnnxClient(modelPath string) (*OnnxInferenceClient, error) {
	// Initialize ONNX Runtime
	// We need to point to the shared library.
	// For this test, we assume the user has set it up or we use the default location if possible.

	// Check if we need to set shared lib path
	// ort.SetSharedLibraryPath(...)

	_ = ort.InitializeEnvironment()
	// if err != nil {
	// 	return nil, fmt.Errorf("failed to init ort: %w", err)
	// }

	options, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}

	// Set intra-op threads to 1 to avoid contention since we have many workers
	options.SetIntraOpNumThreads(1)
	options.SetInterOpNumThreads(1)

	// Create session
	inputs := []string{"input"}
	outputs := []string{"policy", "value"}

	session, err := ort.NewDynamicAdvancedSession(modelPath, inputs, outputs, options)
	if err != nil {
		return nil, fmt.Errorf("failed to create session: %w", err)
	}

	return &OnnxInferenceClient{
		session: session,
	}, nil
}

func (c *OnnxInferenceClient) Predict(state *game.GameState) ([]float32, []float32, error) {
	// Convert state to tensor data
	byteDataPtr := convert.StateToBytes(state)
	byteData := *byteDataPtr
	defer convert.PutBuffer(byteDataPtr)

	// Convert bytes to float32 slice
	// Input size: 14 * 11 * 11 = 1694 floats
	floats := make([]float32, 1694)
	for i := 0; i < 1694; i++ {
		bits := uint32(byteData[i*4]) | uint32(byteData[i*4+1])<<8 | uint32(byteData[i*4+2])<<16 | uint32(byteData[i*4+3])<<24
		floats[i] = math.Float32frombits(bits)
	}

	// Create input tensor
	inputShape := []int64{1, 14, 11, 11}
	inputTensor, err := ort.NewTensor(ort.NewShape(inputShape...), floats)
	if err != nil {
		return nil, nil, err
	}
	defer inputTensor.Destroy()

	// Create output tensors
	policyShape := []int64{1, 4}
	policyTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(policyShape...))
	if err != nil {
		return nil, nil, err
	}
	defer policyTensor.Destroy()

	valueShape := []int64{1, 1}
	valueTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(valueShape...))
	if err != nil {
		return nil, nil, err
	}
	defer valueTensor.Destroy()

	err = c.session.Run([]ort.Value{inputTensor}, []ort.Value{policyTensor, valueTensor})
	if err != nil {
		return nil, nil, err
	}

	// Get outputs
	policyData := policyTensor.GetData()
	policy := make([]float32, len(policyData))
	copy(policy, policyData)

	valueData := valueTensor.GetData()
	value := make([]float32, len(valueData))
	copy(value, valueData)

	return policy, value, nil
}

func BenchmarkOnnxInference(b *testing.B) {
	// This benchmark requires the ONNX model and runtime library
	// Correct path relative to executor/mcts
	modelPath := "../../models/snake_net.onnx"
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		b.Skipf("ONNX model not found at %s, skipping benchmark", modelPath)
	}

	// Setup ONNX Runtime (Download lib if needed)
	// This is a bit hacky for a benchmark, but necessary
	libPath := "../../libonnxruntime.so"
	if runtime.GOOS == "linux" {
		// Check if lib exists, if not, we might fail
		ort.SetSharedLibraryPath(libPath)
	}

	client, err := NewOnnxClient(modelPath)
	if err != nil {
		b.Fatalf("Failed to create ONNX client: %v", err)
	}

	// Create state
	state := &game.GameState{
		Width:  11,
		Height: 11,
		YouId:  "me",
		Snakes: []game.Snake{
			{Id: "me", Health: 100, Body: []game.Point{{X: 5, Y: 5}}},
		},
		Food: []game.Point{{X: 8, Y: 8}},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, err := client.Predict(state)
		if err != nil {
			b.Fatalf("Prediction failed: %v", err)
		}
	}
}
