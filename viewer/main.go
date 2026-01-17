package main

import (
	"flag"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/brensch/snek2/executor/inference"

	_ "github.com/duckdb/duckdb-go/v2"
)

func main() {
	fs := flag.NewFlagSet(os.Args[0], flag.ContinueOnError)
	fs.SetOutput(os.Stderr)

	listen := fs.String("listen", "127.0.0.1:8080", "HTTP listen address")
	dataDir := fs.String("data-dir", "", "Directory containing archive parquet shards (archive_turn_v1) [deprecated: prefer -data-dirs]")
	dataDirs := fs.String("data-dirs", strings.Join(defaultDataDirs(), ","), "Comma-separated list of directories containing archive parquet shards (archive_turn_v1)")
	modelPath := fs.String("model-path", filepath.Join("models", "snake_net.onnx"), "ONNX model path for MCTS explorer")
	mctsSessions := fs.Int("mcts-sessions", 1, "Number of ONNX sessions for MCTS explorer")
	mctsCUDA := fs.Bool("mcts-cuda", true, "Enable CUDA execution provider for MCTS explorer (requires CUDA runtime libs)")
	if err := fs.Parse(os.Args[1:]); err != nil {
		log.Fatalf("flag parse: %v", err)
	}

	roots := parseDataRoots(*dataDirs)
	if strings.TrimSpace(*dataDir) != "" {
		// Back-compat: if user explicitly sets -data-dir, use only that.
		roots = []string{strings.TrimSpace(*dataDir)}
	}

	log.Printf("Viewer data roots: %s", strings.Join(roots, ","))

	var poolOnce sync.Once
	var pool *inference.OnnxPool
	var poolErr error
	getPool := func() (*inference.OnnxPool, error) {
		poolOnce.Do(func() {
			if !*mctsCUDA {
				if os.Getenv("SNEK2_ORT_DISABLE_CUDA") == "" {
					_ = os.Setenv("SNEK2_ORT_DISABLE_CUDA", "1")
				}
			}
			pool, poolErr = inference.NewOnnxClientPool(*modelPath, *mctsSessions)
		})
		return pool, poolErr
	}

	server := NewServer(roots, getPool)
	mux := http.NewServeMux()
	server.RegisterRoutes(mux)

	srv := &http.Server{
		Addr:              *listen,
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
	}
	log.Printf("Viewer API listening on http://%s", *listen)
	log.Fatal(srv.ListenAndServe())
}

func defaultDataDirs() []string {
	// Prefer processed outputs if present.
	preferred := []string{
		filepath.Join("processed", "generated"),
		filepath.Join("processed", "scraped"),
	}
	out := make([]string, 0, len(preferred))
	for _, p := range preferred {
		if _, err := os.Stat(p); err == nil {
			out = append(out, p)
		}
	}
	if len(out) == 0 {
		// Backstop for older layouts.
		out = append(out, filepath.Join("data", "generated"))
	}
	return out
}

func parseDataRoots(csv string) []string {
	parts := strings.Split(csv, ",")
	out := make([]string, 0, len(parts))
	seen := make(map[string]bool, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		if seen[p] {
			continue
		}
		seen[p] = true
		out = append(out, p)
	}
	return out
}
