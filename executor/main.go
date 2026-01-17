package main

import (
	"compress/gzip"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"path/filepath"
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
	rows []store.ArchiveTurnRow
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
	maxGames := fs.Int64("max-games", 0, "If > 0, stop after completing this many games (across all workers). In-flight games are checkpointed for resume.")
	inflightPath := fs.String("inflight-path", "", "Path to checkpoint file for in-flight selfplay games (default: <out-dir>/tmp/inflight_selfplay.json.gz)")
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
		// Default to the symlink which training updates to point to the latest model
		modelPath = "models/snake_net.onnx"
	}
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		log.Fatalf("Model file not found: %s. Run `make export-onnx` first.", modelPath)
	}

	// Resolve symlink to get the actual model file path
	resolvedModelPath := modelPath
	if resolved, err := filepath.EvalSymlinks(modelPath); err == nil {
		resolvedModelPath = resolved
	}

	log.Printf(
		"Config: workers=%d mcts_sims=%d onnx_sessions=%d onnx_batch_size=%d onnx_batch_timeout=%s model=%s (resolved: %s)",
		*workers,
		*mctsSims,
		*onnxSessions,
		*onnxBatchSize,
		(*onnxBatchTimeout).String(),
		modelPath,
		resolvedModelPath,
	)
	// Args logging is noisy and not useful for normal runs.

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

	if strings.TrimSpace(*inflightPath) == "" {
		*inflightPath = filepath.Join(*outDir, "tmp", "inflight_selfplay.json.gz")
	}
	claimedPath := *inflightPath + ".claimed"

	// Claim and load any previous in-flight games so we can resume them.
	// We keep the claimed file around until successful shutdown, so crashes don't lose it.
	if _, err := os.Stat(*inflightPath); err == nil {
		_ = os.MkdirAll(filepath.Dir(*inflightPath), 0o755)
		_ = os.Rename(*inflightPath, claimedPath)
	}

	resumeGames := make([]selfplay.InProgressGame, 0)
	if loaded, err := loadInflightGames(claimedPath); err != nil {
		log.Printf("Failed to load inflight checkpoint %s: %v", claimedPath, err)
	} else if len(loaded) > 0 {
		resumeGames = append(resumeGames, loaded...)
		log.Printf("Loaded %d in-flight games from %s", len(loaded), claimedPath)
	}
	if loaded, err := loadInflightGames(*inflightPath); err != nil {
		log.Printf("Failed to load inflight checkpoint %s: %v", *inflightPath, err)
	} else if len(loaded) > 0 {
		resumeGames = append(resumeGames, loaded...)
		log.Printf("Loaded %d in-flight games from %s", len(loaded), *inflightPath)
	}

	resumeQueue := make(chan selfplay.InProgressGame, len(resumeGames))
	for i := range resumeGames {
		resumeQueue <- resumeGames[i]
	}

	var completedGames atomic.Int64
	var stopRequested atomic.Bool
	shouldStop := func() bool {
		if stopRequested.Load() {
			return true
		}
		if *maxGames > 0 && completedGames.Load() >= *maxGames {
			return true
		}
		return false
	}

	inflightCh := make(chan *selfplay.InProgressGame, (*workers)*2)

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
			doTrace := *trace && workerId == 0
			for {
				select {
				case <-ctx.Done():
					stopRequested.Store(true)
					return
				default:
				}

				if shouldStop() {
					return
				}

				// Prefer resuming a previously checkpointed game.
				var resume *selfplay.InProgressGame
				select {
				case g := <-resumeQueue:
					gg := g
					resume = &gg
				default:
				}

				// Run one game (or checkpoint a resumable chunk)
				onStep := func() {
					totalMoves.Add(1)
				}
				out := selfplay.PlayGameWithOptions(
					ctx,
					workerId,
					mcts.Config{Cpuct: 1.0},
					c,
					doTrace,
					*mctsSims,
					onStep,
					selfplay.PlayGameOptions{
						Resume:        resume,
						StopRequested: func() bool { return shouldStop() },
						ModelPath:     resolvedModelPath,
					},
				)

				if out.Completed {
					total := completedGames.Add(1)
					totalGames.Store(total)
					if *maxGames > 0 {
						log.Printf("Game finished: %d/%d", total, *maxGames)
					}
					if len(out.Rows) > 0 {
						writeReqs <- gameWriteRequest{rows: out.Rows}
						// Keep the updates channel for future UI; don't spam logs per game.
						select {
						case updates <- GameUpdate{WorkerID: workerId, Result: out.Result, Examples: len(out.Rows)}:
						default:
						}
					}
					if *maxGames > 0 && total >= *maxGames {
						stopRequested.Store(true)
						return
					}
					continue
				}

				if out.Checkpoint != nil {
					select {
					case inflightCh <- out.Checkpoint:
					default:
						log.Printf("inflight checkpoint channel full; dropping game %s", out.Checkpoint.GameID)
					}
					return
				}

				log.Printf("Worker %d: Game Aborted (Error)", workerId)
				return
			}
		}(i)
	}

	workersDone := make(chan struct{})
	go func() {
		workerWG.Wait()
		close(inflightCh)
		close(workersDone)
	}()

	// p := tea.NewProgram(initialModel(updates), tea.WithAltScreen())
	// if _, err := p.Run(); err != nil {
	// 	log.Fatal(err)
	// }

	// Temporary replacement for TUI to debug errors
	startTime := time.Now()
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			<-workersDone
			inflight := collectInflight(inflightCh)
			if err := saveOrClearInflight(*inflightPath, inflight); err != nil {
				log.Printf("Failed to persist inflight games: %v", err)
			}
			_ = os.Remove(claimedPath)
			close(writeReqs)
			<-writerDone
			return
		case <-workersDone:
			// Normal completion (e.g., -max-games reached).
			inflight := collectInflight(inflightCh)
			if err := saveOrClearInflight(*inflightPath, inflight); err != nil {
				log.Printf("Failed to persist inflight games: %v", err)
			}
			_ = os.Remove(claimedPath)
			close(writeReqs)
			<-writerDone
			return
		case <-updates:
			// Drain updates silently (kept for potential UI).
		case <-ticker.C:
			duration := time.Since(startTime)
			moves := totalMoves.Load()
			inferences := totalInferences.Load()
			movesPerSec := float64(moves) / duration.Seconds()
			infPerSec := float64(inferences) / duration.Seconds()
			games := completedGames.Load()
			if sp, ok := statsProvider.(interface{ Stats() inference.RuntimeStats }); ok {
				st := sp.Stats()
				log.Printf(
					"Stats: games=%d moves/s=%.0f inf/s=%.0f q=%d batch(avg=%.1f last=%d) run_avg=%.1fms",
					games,
					movesPerSec,
					infPerSec,
					st.QueueLen,
					st.AvgBatchSize,
					st.LastBatchSize,
					st.AvgRunMs,
				)
			} else {
				log.Printf("Stats: games=%d moves/s=%.0f inf/s=%.0f", games, movesPerSec, infPerSec)
			}
		}
	}
}

func parquetWriterLoop(outDir string, gamesPerFlush int, in <-chan gameWriteRequest) {
	if gamesPerFlush <= 0 {
		gamesPerFlush = 50
	}

	pendingRows := make([]store.ArchiveTurnRow, 0, 256*gamesPerFlush)
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

		outPath, err := store.WriteArchiveBatchParquetAtomic(outDir, pendingRows)
		if err != nil {
			log.Printf("Parquet flush failed (games=%d rows=%d): %v", pendingGames, len(pendingRows), err)
		} else {
			_ = outPath
		}

		pendingRows = pendingRows[:0]
		pendingGames = 0
	}

	if pendingGames > 0 && len(pendingRows) > 0 {
		outPath, err := store.WriteArchiveBatchParquetAtomic(outDir, pendingRows)
		if err != nil {
			log.Printf("Parquet final flush failed (games=%d rows=%d): %v", pendingGames, len(pendingRows), err)
			return
		}
		_ = outPath
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

type inflightCheckpointFile struct {
	Version int                       `json:"version"`
	Games   []selfplay.InProgressGame `json:"games"`
}

func loadInflightGames(path string) ([]selfplay.InProgressGame, error) {
	if strings.TrimSpace(path) == "" {
		return nil, nil
	}
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	defer f.Close()

	gr, err := gzip.NewReader(f)
	if err != nil {
		return nil, err
	}
	defer gr.Close()

	var ck inflightCheckpointFile
	if err := json.NewDecoder(gr).Decode(&ck); err != nil {
		return nil, err
	}
	return ck.Games, nil
}

func saveOrClearInflight(path string, games []selfplay.InProgressGame) error {
	if strings.TrimSpace(path) == "" {
		return nil
	}
	if len(games) == 0 {
		_ = os.Remove(path)
		return nil
	}

	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}

	tmpPath := path + ".tmp"
	_ = os.Remove(tmpPath)
	f, err := os.Create(tmpPath)
	if err != nil {
		return err
	}

	gw := gzip.NewWriter(f)
	enc := json.NewEncoder(gw)
	enc.SetIndent("", "  ")
	err = enc.Encode(inflightCheckpointFile{Version: 1, Games: games})
	_ = gw.Close()
	_ = f.Close()
	if err != nil {
		_ = os.Remove(tmpPath)
		return err
	}
	if err := os.Rename(tmpPath, path); err != nil {
		_ = os.Remove(tmpPath)
		return err
	}
	return nil
}

func collectInflight(in <-chan *selfplay.InProgressGame) []selfplay.InProgressGame {
	out := make([]selfplay.InProgressGame, 0)
	for g := range in {
		if g == nil || g.State == nil || g.GameID == "" {
			continue
		}
		out = append(out, *g)
	}
	return out
}
