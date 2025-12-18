package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"sync/atomic"
	"time"

	pb "github.com/brensch/snek2/gen/go"
	"github.com/brensch/snek2/go-worker/mcts"
	"github.com/brensch/snek2/go-worker/selfplay"
	tea "github.com/charmbracelet/bubbletea"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/proto"
)

var totalMoves atomic.Int64
var totalInferences atomic.Int64

type instrumentedClient struct {
	pb.InferenceServiceClient
}

func (c *instrumentedClient) Predict(ctx context.Context, in *pb.InferenceRequest, opts ...grpc.CallOption) (*pb.BatchInferenceResponse, error) {
	totalInferences.Add(1)
	return c.InferenceServiceClient.Predict(ctx, in, opts...)
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
	f, err := os.OpenFile("worker.log", os.O_RDWR|os.O_CREATE|os.O_APPEND, 0666)
	if err != nil {
		log.Fatalf("error opening file: %v", err)
	}
	defer f.Close()
	log.SetOutput(f)

	// Use Unix Domain Socket
	conn, err := grpc.NewClient("unix:///tmp/snek.sock", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("did not connect: %v", err)
	}
	defer conn.Close()
	c := pb.NewInferenceServiceClient(conn)
	c = &instrumentedClient{InferenceServiceClient: c}

	// Wait for server to be ready
	fmt.Println("Waiting for Python Inference Server...")
	for {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		_, err := c.Predict(ctx, &pb.InferenceRequest{})
		cancel()
		if err == nil {
			break
		}
		time.Sleep(500 * time.Millisecond)
	}
	fmt.Println("Server connected! Starting workers...")

	// We use Parallel Games now.
	// Each worker runs sequential MCTS.
	// The Inference Server batches requests from all workers.
	// We use 4096 workers to perfectly fill the GPU batch size.
	workers := 4096
	log.Printf("Starting Self-Play with %d workers", workers)

	updates := make(chan GameUpdate, workers)

	for i := 0; i < workers; i++ {
		go func(workerId int) {
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

	p := tea.NewProgram(initialModel(updates), tea.WithAltScreen())
	if _, err := p.Run(); err != nil {
		log.Fatal(err)
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
