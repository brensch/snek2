package main

import (
	"context"
	"log"
	"sync"
	"time"

	pb "github.com/brensch/snek2/gen/go"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

func main() {
	conn, err := grpc.NewClient("localhost:50051", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("did not connect: %v", err)
	}
	defer conn.Close()
	c := pb.NewInferenceServiceClient(conn)

	// Create a dummy request
	req := &pb.InferenceRequest{
		States: []*pb.GameState{
			{
				Width:  11,
				Height: 11,
				Snakes: []*pb.Snake{
					{
						Id:     "me",
						Health: 100,
						Body: []*pb.Point{
							{X: 5, Y: 5},
						},
					},
				},
				YouId: "me",
			},
		},
	}

	var wg sync.WaitGroup
	numRequests := 10000

	log.Printf("Sending %d concurrent requests...", numRequests)
	start := time.Now()

	for i := 0; i < numRequests; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()

			_, err := c.Predict(ctx, req)
			if err != nil {
				log.Printf("Request %d failed: %v", id, err)
			}
		}(i)
	}

	wg.Wait()
	elapsed := time.Since(start)
	log.Printf("All %d requests finished in %v", numRequests, elapsed)
}
