package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/brensch/snek2/executor/inference"
	"github.com/brensch/snek2/executor/mcts"
	"github.com/brensch/snek2/executor/selfplay"
	"github.com/brensch/snek2/game"
	"github.com/brensch/snek2/scraper/store"
	tea "github.com/charmbracelet/bubbletea"
)

var totalMoves atomic.Int64
var totalInferences atomic.Int64
var totalGames atomic.Int64

type instrumentedClient struct {
	mcts.Predictor
}

func (c *instrumentedClient) Predict(state *game.GameState) ([]float32, []float32, error) {
	totalInferences.Add(1)
	return c.Predictor.Predict(state)
}

type GameUpdate struct {
	WorkerID int
	Result   selfplay.GameResult
	Examples int
}

type gameWriteRequest struct {
	rows []store.TrainingRow
}

type model struct {
	gamesPlayed   int
	totalExamples int
	moves         int64
	inferences    int64
	startTime     time.Time
	recentGames   []string
	updates       chan GameUpdate
}

func initialModel(updates chan GameUpdate) model {
	return model{
		startTime: time.Now(),
		updates:   updates,
	}
}

type TickMsg time.Time

func tickCmd() tea.Cmd {
	return tea.Tick(time.Millisecond*100, func(t time.Time) tea.Msg {
		return TickMsg(t)
	})
}

func (m model) Init() tea.Cmd {
	return tea.Batch(waitForUpdate(m.updates), tickCmd())
}

func waitForUpdate(updates chan GameUpdate) tea.Cmd {
	return func() tea.Msg {
		return <-updates
	}
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		if msg.String() == "q" || msg.String() == "ctrl+c" {
			return m, tea.Quit
		}
	case TickMsg:
		m.moves = totalMoves.Load()
		m.inferences = totalInferences.Load()
		return m, tickCmd()
	case GameUpdate:
		m.gamesPlayed++
		m.totalExamples += msg.Examples
		logMsg := fmt.Sprintf("Worker %d: Winner %s, Steps %d, Ex %d", msg.WorkerID, msg.Result.WinnerId, msg.Result.Steps, msg.Examples)
		m.recentGames = append([]string{logMsg}, m.recentGames...)
		if len(m.recentGames) > 10 {
			m.recentGames = m.recentGames[:10]
		}
		return m, waitForUpdate(m.updates)
	}
	return m, nil
}

func (m model) View() string {
	duration := time.Since(m.startTime)
	gamesPerSec := float64(m.gamesPlayed) / duration.Seconds()
	movesPerSec := float64(m.moves) / duration.Seconds()
	inferencesPerSec := float64(m.inferences) / duration.Seconds()
	if duration.Seconds() < 1 {
		gamesPerSec = 0
		movesPerSec = 0
		inferencesPerSec = 0
	}

	s := fmt.Sprintf("Games Played:   %d\n", m.gamesPlayed)
	s += fmt.Sprintf("Total Examples: %d\n", m.totalExamples)
	s += fmt.Sprintf("Total Moves:    %d\n", m.moves)
	s += fmt.Sprintf("Total Inferences: %d\n", m.inferences)
	s += fmt.Sprintf("Duration:       %s\n", duration.Round(time.Second))
	s += fmt.Sprintf("Games/Sec:      %.2f\n", gamesPerSec)
	s += fmt.Sprintf("Moves/Sec:      %.2f\n", movesPerSec)
	s += fmt.Sprintf("Inferences/Sec: %.2f\n\n", inferencesPerSec)

	s += "Recent Games:\n"
	for _, g := range m.recentGames {
		s += g + "\n"
	}

	s += "\nPress q to quit.\n"
	return s
}

func main() {
	// Use a dedicated FlagSet to avoid any imported package parsing
	// the global flag.CommandLine before main() runs.
	fs := flag.NewFlagSet(os.Args[0], flag.ContinueOnError)
	fs.SetOutput(os.Stderr)

	outDir := fs.String("out-dir", "data/generated", "Output directory for generated training parquet batches")
	workers := fs.Int("workers", 128, "Number of self-play workers")
	gamesPerFlush := fs.Int("games-per-flush", 50, "Number of games to buffer per parquet flush")
	maxGames := fs.Int64("max-games", 0, "If > 0, stop after generating this many games (across all workers)")
	_ = runtime.NumCPU() // keep runtime import useful if you later tune defaults
	onnxSessions := fs.Int("onnx-sessions", 1, "Number of ONNX Runtime sessions to run in parallel (each has its own batching loop). Start with 1; increase if GPU is underutilized.")
	onnxBatchSize := fs.Int("onnx-batch-size", inference.DefaultBatchSize, "ONNX inference batch size (larger can improve GPU utilization)")
	onnxBatchTimeout := fs.Duration("onnx-batch-timeout", inference.DefaultBatchTimeout, "Max time to wait for filling an ONNX batch")
	modelPathFlag := fs.String("model", "", "Path to ONNX model file (defaults to models/snake_net_fp16_f32io.onnx if present)")
	trace := fs.Bool("trace", false, "Print a turn-by-turn trace for a single worker (debug)")
	mctsSims := fs.Int("mcts-sims", 800, "Number of MCTS simulations per move (lower=faster, higher=stronger)")

	if err := fs.Parse(os.Args[1:]); err != nil {
		log.Fatalf("Flag parse error: %v", err)
	}

	sigCtx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	ctx, cancel := context.WithCancel(sigCtx)
	defer cancel()

	// Redirect logs to file to avoid messing up TUI
	// f, err := os.OpenFile("worker.log", os.O_RDWR|os.O_CREATE|os.O_APPEND, 0666)
	// if err != nil {
	// 	log.Fatalf("error opening file: %v", err)
	// }
	// defer f.Close()
	// log.SetOutput(f)

	// Initialize ONNX Client
	modelPath := strings.TrimSpace(*modelPathFlag)
	if modelPath == "" {
		// Prefer fp16 compute w/ fp32 I/O (good for Go + tensor cores).
		preferred := "models/snake_net_fp16_f32io.onnx"
		if _, err := os.Stat(preferred); err == nil {
			modelPath = preferred
		} else {
			modelPath = "models/snake_net.onnx"
		}
	}
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		log.Fatalf("Model file not found: %s. Run `make export-onnx` first.", modelPath)
	}

	log.Printf(
		"Config: workers=%d mcts_sims=%d onnx_sessions=%d onnx_batch_size=%d onnx_batch_timeout=%s model=%s",
		*workers,
		*mctsSims,
		*onnxSessions,
		*onnxBatchSize,
		(*onnxBatchTimeout).String(),
		modelPath,
	)
	log.Printf("Args: %q", os.Args)

	var predictor mcts.Predictor
	var closer interface{ Close() error }
	var statsProvider any
	onnxCfg := inference.OnnxClientConfig{BatchSize: *onnxBatchSize, BatchTimeout: *onnxBatchTimeout}
	if *onnxSessions <= 1 {
		onnxClient, err := inference.NewOnnxClientWithConfig(modelPath, onnxCfg)
		if err != nil {
			log.Fatalf("Failed to create ONNX client: %v", err)
		}
		predictor = onnxClient
		closer = onnxClient
		statsProvider = onnxClient
	} else {
		pool, err := inference.NewOnnxClientPoolWithConfig(modelPath, *onnxSessions, onnxCfg)
		if err != nil {
			log.Fatalf("Failed to create ONNX client pool: %v", err)
		}
		predictor = pool
		closer = pool
		statsProvider = pool
	}
	defer func() {
		if closer != nil {
			_ = closer.Close()
		}
	}()

	// Wrap with instrumentation
	var c mcts.Predictor = &instrumentedClient{Predictor: predictor}

	fmt.Println("ONNX Client initialized! Starting workers...")

	// We use Parallel Games.
	// Each worker runs sequential MCTS with local inference.
	// 16 workers for 16 threads.
	log.Printf("Starting Self-Play with %d workers", *workers)
	// In this self-play setup there are 2 snakes per game; each snake's MCTS runs sequentially
	// and can only have ~1 inference request in flight. So total in-flight requests is roughly
	// workers * 2. If batch size is larger than that, you'll almost never fill batches.
	maxInflight := (*workers) * 2
	if *onnxBatchSize > maxInflight {
		log.Printf("NOTE: onnx-batch-size=%d > max in-flight (~workers*2=%d). Effective batch will cap near %d; consider increasing -workers or lowering -onnx-batch-size.", *onnxBatchSize, maxInflight, maxInflight)
	}

	updates := make(chan GameUpdate, *workers)
	writeReqs := make(chan gameWriteRequest, (*workers)*4)

	// If max-games is set, we must prevent more than that many games from
	// starting concurrently across workers. The previous approach cancelled the
	// context when the *completed* count reached max-games, which caused many
	// in-flight games to abort and still be counted. Here we cap the number of
	// games started and allow in-flight games to finish naturally.
	var startedGames atomic.Int64
	acquireGameStartSlot := func() bool {
		if *maxGames <= 0 {
			return true
		}
		for {
			cur := startedGames.Load()
			if cur >= *maxGames {
				return false
			}
			if startedGames.CompareAndSwap(cur, cur+1) {
				return true
			}
		}
	}

	writerDone := make(chan struct{})
	go func() {
		parquetWriterLoop(*outDir, *gamesPerFlush, writeReqs)
		close(writerDone)
	}()

	var workerWG sync.WaitGroup

	for i := 0; i < *workers; i++ {
		workerWG.Add(1)
		go func(workerId int) {
			defer workerWG.Done()
			log.Printf("Worker %d started", workerId)
			doTrace := *trace && workerId == 0
			for {
				select {
				case <-ctx.Done():
					return
				default:
				}

				if !acquireGameStartSlot() {
					return
				}

				// Run one game
				// Disable verbose for TUI
				onStep := func() {
					totalMoves.Add(1)
				}
				examples, result := selfplay.PlayGame(ctx, workerId, mcts.Config{Cpuct: 1.0}, c, doTrace, *mctsSims, onStep)
				total := totalGames.Add(1)
				log.Printf("finished game number %d", total)

				if len(examples) > 0 {
					writeReqs <- gameWriteRequest{rows: examples}

					// Avoid blocking shutdown if the UI loop stops consuming.
					select {
					case updates <- GameUpdate{WorkerID: workerId, Result: result, Examples: len(examples)}:
					default:
					}

				} else {
					log.Printf("Worker %d: Game Aborted (Error)", workerId)
				}
			}
		}(i)
	}

	workersDone := make(chan struct{})
	go func() {
		workerWG.Wait()
		close(workersDone)
	}()

	// p := tea.NewProgram(initialModel(updates), tea.WithAltScreen())
	// if _, err := p.Run(); err != nil {
	// 	log.Fatal(err)
	// }

	// Temporary replacement for TUI to debug errors
	startTime := time.Now()
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			log.Printf("Shutdown requested; waiting for workers to finish current games...")
			<-workersDone
			close(writeReqs)
			<-writerDone
			log.Printf("Shutdown complete: final parquet flush done (games=%d)", totalGames.Load())
			return
		case <-workersDone:
			// Normal completion (e.g., -max-games reached).
			close(writeReqs)
			<-writerDone
			log.Printf("Shutdown complete: final parquet flush done (games=%d)", totalGames.Load())
			return
		case update := <-updates:
			log.Printf("Worker %d: Winner %s, Steps %d, Ex %d", update.WorkerID, update.Result.WinnerId, update.Result.Steps, update.Examples)
		case <-ticker.C:
			duration := time.Since(startTime)
			moves := totalMoves.Load()
			inferences := totalInferences.Load()
			movesPerSec := float64(moves) / duration.Seconds()
			infPerSec := float64(inferences) / duration.Seconds()
			if sp, ok := statsProvider.(interface{ Stats() inference.RuntimeStats }); ok {
				st := sp.Stats()
				log.Printf("Stats: Moves/s: %.2f, Inf/s: %.2f | batch cfg=%d timeout=%.2fms avg=%.1f last=%d q=%d run avg=%.2fms", movesPerSec, infPerSec, st.ConfigBatchSize, st.ConfigBatchTimeoutMs, st.AvgBatchSize, st.LastBatchSize, st.QueueLen, st.AvgRunMs)
			} else {
				log.Printf("Stats: Moves/s: %.2f, Inf/s: %.2f", movesPerSec, infPerSec)
			}
		}
	}
}

func parquetWriterLoop(outDir string, gamesPerFlush int, in <-chan gameWriteRequest) {
	if gamesPerFlush <= 0 {
		gamesPerFlush = 50
	}

	pendingRows := make([]store.TrainingRow, 0, 1024*gamesPerFlush)
	pendingGames := 0

	for req := range in {
		if len(req.rows) == 0 {
			continue
		}
		pendingRows = append(pendingRows, req.rows...)
		pendingGames++

		if pendingGames < gamesPerFlush {
			continue
		}

		outPath, err := store.WriteBatchParquetAtomic(outDir, pendingRows)
		if err != nil {
			log.Printf("Parquet flush failed (games=%d rows=%d): %v", pendingGames, len(pendingRows), err)
		} else {
			log.Printf("Parquet flush ok: %s (games=%d rows=%d)", outPath, pendingGames, len(pendingRows))
		}

		pendingRows = pendingRows[:0]
		pendingGames = 0
	}

	if pendingGames > 0 && len(pendingRows) > 0 {
		outPath, err := store.WriteBatchParquetAtomic(outDir, pendingRows)
		if err != nil {
			log.Printf("Parquet final flush failed (games=%d rows=%d): %v", pendingGames, len(pendingRows), err)
			return
		}
		log.Printf("Parquet final flush ok: %s (games=%d rows=%d)", outPath, pendingGames, len(pendingRows))
	}
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
