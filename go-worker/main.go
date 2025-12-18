package main

import (
	"log"
	"sync"

	pb "github.com/brensch/snek2/gen/go"
	"github.com/brensch/snek2/go-worker/mcts"
	"github.com/brensch/snek2/go-worker/selfplay"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

func main() {
	// Use Unix Domain Socket
	conn, err := grpc.NewClient("unix:///tmp/snek.sock", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("did not connect: %v", err)
	}
	defer conn.Close()
	c := pb.NewInferenceServiceClient(conn)

	// We use Batched MCTS now, so each worker sends a full batch (256).
	// We only need 1 worker to saturate the GPU and visualize the game.
	workers := 1
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
				} else {
					log.Printf("Worker %d: Game Aborted (Error)", workerId)
				}

				// In the future, we would send 'examples' to a training server or save to disk.
			}
		}(i)
	}

	wg.Wait()
}
