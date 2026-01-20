// Package main implements a Battlesnake API server using neural network inference.
//
// This server responds to the Battlesnake API endpoints and uses MCTS with
// neural network inference to determine the best move within the 500ms time limit.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/brensch/snek2/executor/inference"
	"github.com/brensch/snek2/executor/mcts"
	"github.com/brensch/snek2/game"
)

// Battlesnake API request/response types

type BattlesnakeInfoResponse struct {
	APIVersion string `json:"apiversion"`
	Author     string `json:"author"`
	Color      string `json:"color"`
	Head       string `json:"head"`
	Tail       string `json:"tail"`
	Version    string `json:"version"`
}

type GameRequest struct {
	Game  Game        `json:"game"`
	Turn  int         `json:"turn"`
	Board Board       `json:"board"`
	You   Battlesnake `json:"you"`
}

type Game struct {
	ID      string  `json:"id"`
	Ruleset Ruleset `json:"ruleset"`
	Map     string  `json:"map"`
	Timeout int     `json:"timeout"`
	Source  string  `json:"source"`
}

type Ruleset struct {
	Name     string          `json:"name"`
	Version  string          `json:"version"`
	Settings RulesetSettings `json:"settings"`
}

type RulesetSettings struct {
	FoodSpawnChance     int `json:"foodSpawnChance"`
	MinimumFood         int `json:"minimumFood"`
	HazardDamagePerTurn int `json:"hazardDamagePerTurn"`
}

type Board struct {
	Height  int           `json:"height"`
	Width   int           `json:"width"`
	Food    []Coord       `json:"food"`
	Hazards []Coord       `json:"hazards"`
	Snakes  []Battlesnake `json:"snakes"`
}

type Battlesnake struct {
	ID             string  `json:"id"`
	Name           string  `json:"name"`
	Health         int     `json:"health"`
	Body           []Coord `json:"body"`
	Latency        string  `json:"latency"`
	Head           Coord   `json:"head"`
	Length         int     `json:"length"`
	Shout          string  `json:"shout"`
	Squad          string  `json:"squad"`
	Customizations struct {
		Color string `json:"color"`
		Head  string `json:"head"`
		Tail  string `json:"tail"`
	} `json:"customizations"`
}

type Coord struct {
	X int `json:"x"`
	Y int `json:"y"`
}

type MoveResponse struct {
	Move  string `json:"move"`
	Shout string `json:"shout,omitempty"`
}

// Server holds the inference client and configuration
type Server struct {
	pool        *inference.OnnxPool
	mctsConfig  mcts.Config
	moveTimeout time.Duration
	mctsSims    int
	mu          sync.RWMutex
}

func NewServer(pool *inference.OnnxPool, moveTimeout time.Duration, mctsSims int) *Server {
	return &Server{
		pool:        pool,
		mctsConfig:  mcts.Config{Cpuct: 1.5},
		moveTimeout: moveTimeout,
		mctsSims:    mctsSims,
	}
}

// handleIndex returns the Battlesnake info
func (s *Server) handleIndex(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}

	response := BattlesnakeInfoResponse{
		APIVersion: "1",
		Author:     "snek2",
		Color:      "#00ff00",
		Head:       "default",
		Tail:       "default",
		Version:    "1.0.0",
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// handleStart is called when a game starts
func (s *Server) handleStart(w http.ResponseWriter, r *http.Request) {
	var req GameRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	log.Printf("Game started: %s, Turn: %d, You: %s", req.Game.ID, req.Turn, req.You.Name)
	w.WriteHeader(http.StatusOK)
}

// handleMove determines the best move using MCTS
func (s *Server) handleMove(w http.ResponseWriter, r *http.Request) {
	startTime := time.Now()

	var req GameRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Convert API request to game state
	state := convertToGameState(&req)

	// Calculate available time for MCTS (leave buffer for response)
	timeout := s.moveTimeout
	if req.Game.Timeout > 0 {
		timeout = time.Duration(req.Game.Timeout) * time.Millisecond
	}
	// Reserve 200ms for overhead and network latency
	computeTime := timeout - 200*time.Millisecond
	if computeTime < 50*time.Millisecond {
		computeTime = 50 * time.Millisecond
	}

	ctx, cancel := context.WithTimeout(r.Context(), computeTime)
	defer cancel()

	// Run MCTS to find the best move
	move, stats := s.runMCTS(ctx, state)

	// Convert move to string
	moveStr := moveToString(move)

	elapsed := time.Since(startTime)
	log.Printf("Turn %d: Move=%s, Sims=%d, Time=%v", req.Turn, moveStr, stats.Simulations, elapsed)

	response := MoveResponse{
		Move:  moveStr,
		Shout: fmt.Sprintf("Ran %d simulations", stats.Simulations),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// handleEnd is called when a game ends
func (s *Server) handleEnd(w http.ResponseWriter, r *http.Request) {
	var req GameRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Check if we won
	youAlive := false
	for _, snake := range req.Board.Snakes {
		if snake.ID == req.You.ID {
			youAlive = true
			break
		}
	}

	result := "lost"
	if youAlive {
		result = "won"
	} else if len(req.Board.Snakes) == 0 {
		result = "draw"
	}

	log.Printf("Game ended: %s, Turn: %d, Result: %s", req.Game.ID, req.Turn, result)
	w.WriteHeader(http.StatusOK)
}

type MCTSStats struct {
	Simulations int
}

// runMCTS performs MCTS within the given context timeout
func (s *Server) runMCTS(ctx context.Context, state *game.GameState) (int, MCTSStats) {
	// Reorder snakes so that our snake (YouId) is first
	// This ensures the MCTS root node is for our snake's decision
	reorderSnakesForYou(state)

	// Run MCTS with the available time
	// The pool is used directly as it implements the Predictor interface
	root, _, err := mcts.SimultaneousSearch(ctx, s.pool, s.mctsConfig, state, s.mctsSims)
	if err != nil && err != context.DeadlineExceeded && err != context.Canceled {
		log.Printf("MCTS error: %v", err)
		return s.fallbackMove(state), MCTSStats{}
	}

	if root == nil || !root.IsExpanded || len(root.Children) == 0 {
		return s.fallbackMove(state), MCTSStats{Simulations: 0}
	}

	// Select the move with the most visits
	bestMove := 0
	bestVisits := -1
	for _, child := range root.Children {
		if child.VisitCount > bestVisits {
			bestVisits = child.VisitCount
			bestMove = child.Move
		}
	}

	return bestMove, MCTSStats{Simulations: root.VisitCount}
}

// reorderSnakesForYou moves the snake matching YouId to the front of the Snakes slice.
// This ensures MCTS builds the tree with our snake's decisions at the root.
func reorderSnakesForYou(state *game.GameState) {
	for i, snake := range state.Snakes {
		if snake.Id == state.YouId && i != 0 {
			// Swap with index 0
			state.Snakes[0], state.Snakes[i] = state.Snakes[i], state.Snakes[0]
			return
		}
	}
}

// fallbackMove returns a safe move when MCTS fails
func (s *Server) fallbackMove(state *game.GameState) int {
	legalMoves := game.GetLegalMoves(state)
	if len(legalMoves) == 0 {
		return game.MoveUp // No legal moves, return up (will die anyway)
	}
	return legalMoves[0]
}

// convertToGameState converts a Battlesnake API request to our game state
func convertToGameState(req *GameRequest) *game.GameState {
	state := &game.GameState{
		Width:  int32(req.Board.Width),
		Height: int32(req.Board.Height),
		YouId:  req.You.ID,
		Turn:   int32(req.Turn),
	}

	// Convert food
	state.Food = make([]game.Point, len(req.Board.Food))
	for i, f := range req.Board.Food {
		state.Food[i] = game.Point{X: int32(f.X), Y: int32(f.Y)}
	}

	// Convert snakes
	state.Snakes = make([]game.Snake, len(req.Board.Snakes))
	for i, s := range req.Board.Snakes {
		snake := game.Snake{
			Id:     s.ID,
			Health: int32(s.Health),
			Body:   make([]game.Point, len(s.Body)),
		}
		for j, b := range s.Body {
			snake.Body[j] = game.Point{X: int32(b.X), Y: int32(b.Y)}
		}
		state.Snakes[i] = snake
	}

	return state
}

// moveToString converts a move index to a string
func moveToString(move int) string {
	switch move {
	case game.MoveUp:
		return "up"
	case game.MoveDown:
		return "down"
	case game.MoveLeft:
		return "left"
	case game.MoveRight:
		return "right"
	default:
		return "up"
	}
}

func main() {
	fs := flag.NewFlagSet(os.Args[0], flag.ContinueOnError)
	fs.SetOutput(os.Stderr)

	listen := fs.String("listen", ":8080", "HTTP listen address")
	modelPath := fs.String("model-path", filepath.Join("models", "snake_net.onnx"), "Path to ONNX model")
	sessions := fs.Int("sessions", 1, "Number of ONNX sessions (for parallel games)")
	moveTimeout := fs.Duration("move-timeout", 500*time.Millisecond, "Default move timeout")
	mctsSims := fs.Int("mcts-sims", 10000, "Max MCTS simulations per move (will stop early if timeout)")
	disableCUDA := fs.Bool("disable-cuda", false, "Disable CUDA execution provider")

	if err := fs.Parse(os.Args[1:]); err != nil {
		log.Fatalf("flag parse: %v", err)
	}

	// Set CUDA disable flag if requested
	if *disableCUDA {
		if os.Getenv("SNEK2_ORT_DISABLE_CUDA") == "" {
			_ = os.Setenv("SNEK2_ORT_DISABLE_CUDA", "1")
		}
	}

	// Ensure LD_LIBRARY_PATH includes necessary paths for CUDA libs
	ensureLibraryPath()

	log.Printf("Loading model from: %s", *modelPath)
	log.Printf("CUDA enabled: %v", !*disableCUDA && os.Getenv("SNEK2_ORT_DISABLE_CUDA") == "")

	// Create inference pool
	pool, err := inference.NewOnnxClientPool(*modelPath, *sessions)
	if err != nil {
		log.Fatalf("Failed to create inference pool: %v", err)
	}

	server := NewServer(pool, *moveTimeout, *mctsSims)

	mux := http.NewServeMux()
	mux.HandleFunc("/", server.handleIndex)
	mux.HandleFunc("/start", server.handleStart)
	mux.HandleFunc("/move", server.handleMove)
	mux.HandleFunc("/end", server.handleEnd)

	srv := &http.Server{
		Addr:              *listen,
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
	}

	log.Printf("Battlesnake server listening on http://%s", *listen)
	log.Fatal(srv.ListenAndServe())
}

// ensureLibraryPath sets up LD_LIBRARY_PATH for CUDA libraries
func ensureLibraryPath() {
	cwd, err := os.Getwd()
	if err != nil {
		return
	}

	// Find the repo root by looking for libonnxruntime.so.1
	repoDir := cwd
	for up := 0; up < 8; up++ {
		if _, err := os.Stat(filepath.Join(repoDir, "libonnxruntime.so.1")); err == nil {
			break
		}
		parent := filepath.Dir(repoDir)
		if parent == repoDir {
			break
		}
		repoDir = parent
	}

	// Build list of paths to add
	candidateDirs := []string{cwd}
	if repoDir != "" && repoDir != cwd {
		candidateDirs = append(candidateDirs, repoDir)
	}

	// Add .venv library paths for CUDA
	patterns := []string{
		filepath.Join(repoDir, ".venv", "lib", "python*", "site-packages", "nvidia", "*", "lib"),
		filepath.Join(repoDir, ".venv", "lib", "python*", "site-packages", "triton", "backends", "nvidia", "lib"),
		filepath.Join(repoDir, ".venv", "lib", "python*", "site-packages", "torch", "lib"),
	}
	for _, pat := range patterns {
		matches, _ := filepath.Glob(pat)
		candidateDirs = append(candidateDirs, matches...)
	}

	existing := os.Getenv("LD_LIBRARY_PATH")
	existingSet := map[string]bool{}
	for _, p := range strings.Split(existing, ":") {
		if p != "" {
			existingSet[p] = true
		}
	}

	toAdd := make([]string, 0, len(candidateDirs))
	for _, d := range candidateDirs {
		if existingSet[d] {
			continue
		}
		if st, err := os.Stat(d); err == nil && st.IsDir() {
			toAdd = append(toAdd, d)
		}
	}

	if len(toAdd) == 0 {
		return
	}

	newVal := strings.Join(toAdd, ":")
	if existing != "" {
		newVal = newVal + ":" + existing
	}
	_ = os.Setenv("LD_LIBRARY_PATH", newVal)
}
