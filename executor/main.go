package main

import (
	"fmt"
	"log"
	"os"
	"sync/atomic"
	"time"

	"github.com/brensch/snek2/executor/inference"
	"github.com/brensch/snek2/executor/mcts"
	"github.com/brensch/snek2/executor/selfplay"
	pb "github.com/brensch/snek2/gen/go"
	tea "github.com/charmbracelet/bubbletea"
	"google.golang.org/protobuf/proto"
)

var totalMoves atomic.Int64
var totalInferences atomic.Int64

type instrumentedClient struct {
	mcts.Predictor
}

func (c *instrumentedClient) Predict(state *pb.GameState) ([]float32, []float32, error) {
	totalInferences.Add(1)
	return c.Predictor.Predict(state)
}

type GameUpdate struct {
	WorkerID int
	Result   selfplay.GameResult
	Examples int
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
	// Redirect logs to file to avoid messing up TUI
	// f, err := os.OpenFile("worker.log", os.O_RDWR|os.O_CREATE|os.O_APPEND, 0666)
	// if err != nil {
	// 	log.Fatalf("error opening file: %v", err)
	// }
	// defer f.Close()
	// log.SetOutput(f)

	// Initialize ONNX Client
	modelPath := "models/snake_net.onnx"
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		log.Fatalf("Model file not found: %s. Run trainer/export_onnx.py first.", modelPath)
	}

	onnxClient, err := inference.NewOnnxClient(modelPath)
	if err != nil {
		log.Fatalf("Failed to create ONNX client: %v", err)
	}
	defer onnxClient.Close()

	// Wrap with instrumentation
	var c mcts.Predictor = &instrumentedClient{Predictor: onnxClient}

	fmt.Println("ONNX Client initialized! Starting workers...")

	// We use Parallel Games.
	// Each worker runs sequential MCTS with local inference.
	// 16 workers for 16 threads.
	workers := 128
	log.Printf("Starting Self-Play with %d workers", workers)

	updates := make(chan GameUpdate, workers)

	for i := 0; i < workers; i++ {
		go func(workerId int) {
			log.Printf("Worker %d started", workerId)
			for {
				// Run one game
				// Disable verbose for TUI
				onStep := func() {
					totalMoves.Add(1)
				}
				examples, result := selfplay.PlayGame(workerId, mcts.Config{Cpuct: 1.0}, c, false, onStep)

				if examples != nil {
					if err := saveGame(examples, workerId); err != nil {
						log.Printf("Worker %d: Failed to save game: %v", workerId, err)
					}

					updates <- GameUpdate{
						WorkerID: workerId,
						Result:   result,
						Examples: len(examples),
					}

				} else {
					log.Printf("Worker %d: Game Aborted (Error)", workerId)
				}
			}
		}(i)
	}

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
		case update := <-updates:
			log.Printf("Worker %d: Winner %s, Steps %d, Ex %d", update.WorkerID, update.Result.WinnerId, update.Result.Steps, update.Examples)
		case <-ticker.C:
			duration := time.Since(startTime)
			moves := totalMoves.Load()
			inferences := totalInferences.Load()
			log.Printf("Stats: Moves/s: %.2f, Inf/s: %.2f", float64(moves)/duration.Seconds(), float64(inferences)/duration.Seconds())
		}
	}
}

func saveGame(examples []*pb.TrainingExample, workerId int) error {
	data := &pb.TrainingData{
		Examples: examples,
	}

	bytes, err := proto.Marshal(data)
	if err != nil {
		return err
	}

	filename := fmt.Sprintf("data/game_%d_%d.pb", time.Now().UnixNano(), workerId)
	return os.WriteFile(filename, bytes, 0644)
}
