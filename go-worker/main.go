package main

import (
	"fmt"
	"log"
	"os"
	"sync"
	"time"

	pb "github.com/brensch/snek2/gen/go"
	"github.com/brensch/snek2/go-worker/mcts"
	"github.com/brensch/snek2/go-worker/selfplay"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/proto"
)

func main() {
	// Use Unix Domain Socket
	conn, err := grpc.NewClient("unix:///tmp/snek.sock", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("did not connect: %v", err)
	}
	defer conn.Close()
	c := pb.NewInferenceServiceClient(conn)

	// We use Parallel Games now.
	// Each worker runs sequential MCTS.
	// The Inference Server batches requests from all workers.
	// We use 256 workers to perfectly fill the GPU batch size.
	workers := 256
	log.Printf("Starting Self-Play with %d workers", workers)

	var wg sync.WaitGroup

	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(workerId int) {
			defer wg.Done()
			for {
				// Run one game
				// Only worker 0 is verbose
				verbose := (workerId == 0)
				examples, result := selfplay.PlayGame(workerId, mcts.Config{Cpuct: 1.0}, c, verbose)

				if examples != nil {
					log.Printf("Worker %d: Game Finished. Winner: %s, Steps: %d, Examples: %d",
						workerId, result.WinnerId, result.Steps, len(examples))

					// Save Data
					if err := saveGame(examples, workerId); err != nil {
						log.Printf("Worker %d: Failed to save game: %v", workerId, err)
					}

				} else {
					log.Printf("Worker %d: Game Aborted (Error)", workerId)
				}
			}
		}(i)
	}

	wg.Wait()
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
