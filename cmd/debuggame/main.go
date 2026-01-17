package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"

	"github.com/brensch/snek2/executor/inference"
	"github.com/brensch/snek2/executor/mcts"
	"github.com/brensch/snek2/executor/selfplay"
)

func main() {
	modelPath := flag.String("model", filepath.Join("models", "snake_net.onnx"), "Path to ONNX model")
	outDir := flag.String("out-dir", filepath.Join("debug_games"), "Output directory for debug games")
	sims := flag.Int("sims", 100, "Number of MCTS simulations per move")
	cpuct := flag.Float64("cpuct", 1.0, "MCTS exploration constant")
	cuda := flag.Bool("cuda", true, "Enable CUDA for inference")
	frontendHost := flag.String("frontend", "http://localhost:5173", "Frontend base URL")
	flag.Parse()

	if !*cuda {
		os.Setenv("SNEK2_ORT_DISABLE_CUDA", "1")
	}

	log.Printf("Loading model: %s", *modelPath)
	pool, err := inference.NewOnnxClientPool(*modelPath, 1)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	defer pool.Close()

	mctsConfig := mcts.Config{Cpuct: float32(*cpuct)}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	log.Printf("Generating debug game (unified MCTS) with %d sims, cpuct=%.2f", *sims, *cpuct)

	moveNames := []string{"Up", "Down", "Left", "Right"}
	onProgress := func(p selfplay.DebugProgress) {
		movesStr := ""
		for snakeID, move := range p.Moves {
			moveName := "?"
			if move >= 0 && move < len(moveNames) {
				moveName = moveNames[move]
			}
			if movesStr != "" {
				movesStr += ", "
			}
			movesStr += fmt.Sprintf("%s→%s", snakeID, moveName)
		}
		fmt.Printf("  Turn %3d | %d alive | %s\n", p.Turn, p.AliveCount, movesStr)
	}

	result, err := selfplay.PlayDebugGameV2(ctx, mctsConfig, pool, *modelPath, *sims, onProgress)
	if err != nil {
		log.Fatalf("Failed to generate debug game: %v", err)
	}

	log.Printf("Game complete: %d turns, winner: %s", result.TurnCount, result.Winner)

	// Write to parquet
	parquetPath, err := selfplay.WriteDebugGameParquetV2(*outDir, result)
	if err != nil {
		log.Fatalf("Failed to write debug game: %v", err)
	}

	log.Printf("Debug game written to: %s", parquetPath)

	// Print the frontend URL
	gameID := result.GameID
	frontendURL := fmt.Sprintf("%s/debug/%s", *frontendHost, gameID)

	fmt.Println()
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Printf("  Debug game ready! Open in browser:\n")
	fmt.Printf("  %s\n", frontendURL)
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println()
}
