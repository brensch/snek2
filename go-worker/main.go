package main

import (
	"context"
	"log"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	pb "github.com/brensch/snek2/gen/go"
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

	// Create a dummy GameState
	dummyState := &pb.GameState{
		Width:  11,
		Height: 11,
		YouId:  "me",
		Snakes: []*pb.Snake{
			{
				Id:     "me",
				Health: 90,
				Body: []*pb.Point{
					{X: 5, Y: 5}, // Head
					{X: 5, Y: 4}, // Body
					{X: 5, Y: 3}, // Tail
				},
			},
			{
				Id:     "enemy1",
				Health: 80,
				Body: []*pb.Point{
					{X: 1, Y: 1},
					{X: 1, Y: 2},
				},
			},
		},
		Food: []*pb.Point{
			{X: 8, Y: 8},
		},
	}

	// Load test configuration
	duration := 30 * time.Second
	workers := runtime.NumCPU() * 20 // High concurrency

	var wg sync.WaitGroup
	var requestCount int64
	var errorCount int64

	log.Printf("Starting load test: %d workers (CPUs: %d), %v duration", workers, runtime.NumCPU(), duration)
	start := time.Now()

	ctx, cancel := context.WithTimeout(context.Background(), duration)
	defer cancel()

	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-ctx.Done():
					return
				default:
					// Serialize state (simulating work)
					dataPtr := StateToBytes(dummyState)

					req := &pb.InferenceRequest{
						Data:  *dataPtr,
						Shape: []int32{int32(Channels), int32(Width), int32(Height)},
					}

					// Use a short timeout for individual requests to fail fast if overloaded
					reqCtx, reqCancel := context.WithTimeout(context.Background(), 1*time.Second)
					_, err := c.Predict(reqCtx, req)
					reqCancel()

					PutBuffer(dataPtr)

					if err != nil {
						atomic.AddInt64(&errorCount, 1)
					} else {
						atomic.AddInt64(&requestCount, 1)
					}
				}
			}
		}()
	}

	wg.Wait()
	elapsed := time.Since(start)

	log.Printf("Load test finished in %v", elapsed)
	log.Printf("Total Requests: %d", requestCount)
	log.Printf("Total Errors: %d", errorCount)
	log.Printf("RPS: %.2f", float64(requestCount)/elapsed.Seconds())
}
