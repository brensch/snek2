package inference

import (
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/brensch/snek2/executor/convert"
	"github.com/brensch/snek2/game"
	ort "github.com/yalue/onnxruntime_go"
)

func randomState(r *rand.Rand) *game.GameState {
	// Minimal plausible state for encoding benchmark.
	// (We don't need a fully rules-valid state for encoding perf.)
	w := int32(convert.Width)
	h := int32(convert.Height)

	mkPoint := func() game.Point {
		return game.Point{X: int32(r.Intn(int(w))), Y: int32(r.Intn(int(h)))}
	}

	mkSnake := func(id string, length int, health int32) game.Snake {
		body := make([]game.Point, 0, length)
		p := mkPoint()
		for i := 0; i < length; i++ {
			body = append(body, p)
		}
		return game.Snake{Id: id, Health: health, Body: body}
	}

	s := &game.GameState{}
	s.YouId = "you"
	s.Food = []game.Point{mkPoint(), mkPoint()}
	s.Snakes = []game.Snake{
		mkSnake("you", 3, 90),
		mkSnake("e1", 5, 70),
		mkSnake("e2", 7, 55),
		mkSnake("e3", 9, 30),
	}
	return s
}

func BenchmarkStateToFloat32(b *testing.B) {
	r := rand.New(rand.NewSource(1))
	states := make([]*game.GameState, 1024)
	for i := range states {
		states[i] = randomState(r)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ptr := convert.StateToFloat32(states[i%len(states)])
		convert.PutFloatBuffer(ptr)
	}
}

func BenchmarkOnnxSessionRun(b *testing.B) {
	modelCandidates := []string{
		"../../models/snake_net_fp16_f32io.onnx",
		"../../models/snake_net.onnx",
		"../../models_bak/snake_net.onnx",
	}
	modelPath := ""
	if p := os.Getenv("SNEK2_BENCH_ONNX_MODEL"); p != "" {
		modelCandidates = append([]string{p}, modelCandidates...)
	}
	for _, p := range modelCandidates {
		if _, err := os.Stat(p); err == nil {
			modelPath = p
			break
		}
	}
	if modelPath == "" {
		b.Skip("ONNX model not found in models/ or models_bak/; skipping")
	}
	b.Logf("Using model: %s", modelPath)

	// Configure ORT shared library path similarly to production code.
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
			// When running `go test`, cwd is usually the package dir.
			// Search upwards for the repo root where the lib files live.
			dir := cwd
			for up := 0; up < 6; up++ {
				for _, name := range candidates {
					abs := filepath.Join(dir, name)
					if _, err := os.Stat(abs); err == nil {
						ort.SetSharedLibraryPath(abs)
						up = 999
						break
					}
				}
				parent := filepath.Dir(dir)
				if parent == dir {
					break
				}
				dir = parent
			}
		}
	}

	// Best-effort init; if ORT isn't usable in this environment, skip.
	if err := ort.InitializeEnvironment(); err != nil {
		// onnxruntime_go can return an error if another test already initialized the env.
		if !strings.Contains(err.Error(), "already been initialized") {
			b.Skipf("ORT init failed: %v", err)
		}
	}

	opts, err := ort.NewSessionOptions()
	if err != nil {
		b.Skipf("ORT options failed: %v", err)
	}
	defer opts.Destroy()
	opts.SetIntraOpNumThreads(1)
	opts.SetInterOpNumThreads(1)

	// Try CUDA EP (this is what we care about for "game generation" throughput).
	if disableCUDA := os.Getenv("SNEK2_ORT_DISABLE_CUDA"); disableCUDA != "" && disableCUDA != "0" && strings.ToLower(disableCUDA) != "false" {
		b.Log("CUDA provider disabled via SNEK2_ORT_DISABLE_CUDA")
	} else {
		cudaOpts, err := ort.NewCUDAProviderOptions()
		if err != nil {
			b.Skipf("CUDA provider options unavailable: %v", err)
		}
		defer cudaOpts.Destroy()
		if err := opts.AppendExecutionProviderCUDA(cudaOpts); err != nil {
			b.Skipf("Failed to append CUDA EP (likely missing provider/CUDA libs): %v", err)
		}
		b.Log("CUDA provider enabled")
	}

	inputs := []string{"input"}
	outputs := []string{"policy", "value"}
	sess, err := ort.NewDynamicAdvancedSession(modelPath, inputs, outputs, opts)
	if err != nil {
		b.Skipf("ORT session failed: %v", err)
	}

	batch := int64(128)
	inShape := ort.NewShape(batch, int64(convert.Channels), int64(convert.Height), int64(convert.Width))
	inSize := int(batch) * convert.Channels * convert.Height * convert.Width
	input := make([]float32, inSize)
	for i := range input {
		input[i] = float32(i%7) / 7.0
	}

	inTensor, err := ort.NewTensor(inShape, input)
	if err != nil {
		b.Fatalf("input tensor: %v", err)
	}
	defer inTensor.Destroy()

	policyTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(batch, 4))
	if err != nil {
		b.Fatalf("policy tensor: %v", err)
	}
	defer policyTensor.Destroy()

	valueTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(batch, 1))
	if err != nil {
		b.Fatalf("value tensor: %v", err)
	}
	defer valueTensor.Destroy()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := sess.Run([]ort.Value{inTensor}, []ort.Value{policyTensor, valueTensor}); err != nil {
			b.Fatalf("run: %v", err)
		}
	}
	b.StopTimer()
	dt := b.Elapsed().Seconds()
	if dt > 0 {
		b.ReportMetric((float64(b.N)*float64(batch))/dt, "inf/s")
	}
}
