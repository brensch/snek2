package main

import (
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/brensch/snek2/executor/inference"
	"github.com/brensch/snek2/executor/mcts"
	gamepkg "github.com/brensch/snek2/game"
)

// Server holds shared state for HTTP handlers.
type Server struct {
	roots   []string
	dbCache *DBCache
	getPool func() (*inference.OnnxPool, error)
}

// NewServer creates a new Server with the given data roots and ONNX pool factory.
func NewServer(roots []string, getPool func() (*inference.OnnxPool, error)) *Server {
	return &Server{
		roots:   roots,
		dbCache: NewDBCache(roots, 30*time.Second),
		getPool: getPool,
	}
}

// RegisterRoutes sets up all API routes on the given mux.
func (s *Server) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/api/games", s.handleGames)
	mux.HandleFunc("/api/games/", s.handleGameTurns)
	mux.HandleFunc("/api/stats", s.handleStats)
	mux.HandleFunc("/api/run_inference", s.handleRunInference)
}

func (s *Server) handleGames(w http.ResponseWriter, r *http.Request) {
	withCORS(w, r)
	if r.Method == http.MethodOptions {
		return
	}
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Force DB refresh to ensure we see the latest games from disk.
	if err := s.dbCache.Refresh(); err != nil {
		http.Error(w, fmt.Sprintf("failed to refresh db: %v", err), http.StatusInternalServerError)
		return
	}

	// Use cached games index for fast pagination
	gamesIndex, err := s.dbCache.GetGamesIndex(r.Context())
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	limit := parseIntQuery(r, "limit", 100000)
	offset := parseIntQuery(r, "offset", 0)
	sortKey := strings.TrimSpace(r.URL.Query().Get("sort"))
	sortDir := strings.TrimSpace(r.URL.Query().Get("dir"))

	games, total := queryGamesFromIndex(gamesIndex, limit, offset, sortKey, sortDir)
	writeJSON(w, GamesResponse{Total: total, Games: games})
}

func (s *Server) handleGameTurns(w http.ResponseWriter, r *http.Request) {
	withCORS(w, r)
	if r.Method == http.MethodOptions {
		return
	}
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	db, err := s.dbCache.Get()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// /api/games/{id}/turns
	rest := strings.TrimPrefix(r.URL.Path, "/api/games/")
	parts := strings.Split(rest, "/")
	if len(parts) != 2 || parts[0] == "" || parts[1] != "turns" {
		http.NotFound(w, r)
		return
	}
	gameID, err := url.PathUnescape(parts[0])
	if err != nil {
		http.Error(w, "bad game id", http.StatusBadRequest)
		return
	}
	turns, err := queryTurns(r.Context(), db, gameID)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			http.NotFound(w, r)
			return
		}
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	writeJSON(w, turns)
}

func (s *Server) handleStats(w http.ResponseWriter, r *http.Request) {
	withCORS(w, r)
	if r.Method == http.MethodOptions {
		return
	}
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	fromNs := parseInt64Query(r, "from_ns", 0)
	toNs := parseInt64Query(r, "to_ns", 0)
	bucketNs := parseInt64Query(r, "bucket_ns", 5*60*1_000_000_000)
	if bucketNs <= 0 {
		bucketNs = 5 * 60 * 1_000_000_000
	}
	if fromNs <= 0 || toNs <= 0 || toNs <= fromNs {
		// Default: last 24h.
		nowNs := time.Now().UnixNano()
		toNs = nowNs
		fromNs = nowNs - int64(24*time.Hour)
	}

	db, err := s.dbCache.Get()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	points, err := queryStats(r.Context(), db, fromNs, toNs, bucketNs)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	writeJSON(w, StatsResponse{FromNs: fromNs, ToNs: toNs, BucketNs: bucketNs, Points: points})
}

// RunInferenceRequest is the request body for /api/run_inference.
type RunInferenceRequest struct {
	GameID string `json:"game_id"`
	Turn   int32  `json:"turn"`
	Sims   int    `json:"sims"`
}

// RunInferenceResponse contains the full MCTS tree for visualization.
type RunInferenceResponse struct {
	GameID     string          `json:"game_id"`
	Turn       int32           `json:"turn"`
	Sims       int             `json:"sims"`
	SnakeOrder []string        `json:"snake_order"`
	State      *DebugGameState `json:"state"`
	Tree       *DebugMCTSNode  `json:"tree"`
}

// handleRunInference runs MCTS inference on a specific game turn and returns the full tree.
func (s *Server) handleRunInference(w http.ResponseWriter, r *http.Request) {
	withCORS(w, r)
	if r.Method == http.MethodOptions {
		return
	}
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req RunInferenceRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid request body: "+err.Error(), http.StatusBadRequest)
		return
	}

	if req.GameID == "" {
		http.Error(w, "game_id required", http.StatusBadRequest)
		return
	}
	if req.Sims <= 0 {
		req.Sims = 100 // default
	}
	if req.Sims > 1000 {
		req.Sims = 1000 // cap for safety
	}

	// Get the pool
	pool, err := s.getPool()
	if err != nil {
		http.Error(w, "model not available: "+err.Error(), http.StatusServiceUnavailable)
		return
	}

	// Load the turn from DB
	db, err := s.dbCache.Get()
	if err != nil {
		http.Error(w, "database error: "+err.Error(), http.StatusInternalServerError)
		return
	}

	ctx := r.Context()
	turn, err := queryTurn(ctx, db, req.GameID, req.Turn)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			http.NotFound(w, r)
			return
		}
		http.Error(w, "failed to load turn: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// Convert Turn to game.GameState
	gameState, err := turnToGameStateForMCTS(turn)
	if err != nil {
		http.Error(w, "failed to convert turn: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// Run simultaneous MCTS - alternating tree with shared inference per round
	// All snakes see the same state within a round, no information leakage
	mctsConfig := mcts.Config{Cpuct: 1.0}
	root, snakeOrder, err := mcts.SimultaneousSearch(ctx, pool, mctsConfig, gameState, req.Sims)
	if err != nil {
		http.Error(w, "MCTS search failed: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// Convert to debug format
	debugTree := convertSimultaneousNodeToDebug(root, snakeOrder, mctsConfig.Cpuct)
	debugState := turnToDebugState(&turn)

	resp := RunInferenceResponse{
		GameID:     req.GameID,
		Turn:       req.Turn,
		Sims:       req.Sims,
		SnakeOrder: snakeOrder,
		State:      debugState,
		Tree:       debugTree,
	}

	writeJSON(w, resp)
}

// turnToGameStateForMCTS converts a viewer Turn to a game.GameState for unified MCTS.
func turnToGameStateForMCTS(t Turn) (*gamepkg.GameState, error) {
	if t.Width <= 0 || t.Height <= 0 {
		return nil, fmt.Errorf("invalid board size")
	}

	state := &gamepkg.GameState{
		Turn:   t.Turn,
		Width:  t.Width,
		Height: t.Height,
		Food:   make([]gamepkg.Point, 0, len(t.Food)),
		Snakes: make([]gamepkg.Snake, 0, len(t.Snakes)),
	}

	for _, p := range t.Food {
		state.Food = append(state.Food, gamepkg.Point{X: p.X, Y: p.Y})
	}

	for _, s := range t.Snakes {
		body := make([]gamepkg.Point, 0, len(s.Body))
		for _, bp := range s.Body {
			body = append(body, gamepkg.Point{X: bp.X, Y: bp.Y})
		}
		state.Snakes = append(state.Snakes, gamepkg.Snake{
			Id:     s.ID,
			Health: s.Health,
			Body:   body,
		})
	}

	return state, nil
}

// turnToDebugState converts a viewer Turn to DebugGameState.
func turnToDebugState(t *Turn) *DebugGameState {
	snakes := make([]DebugSnakeState, 0, len(t.Snakes))
	for _, s := range t.Snakes {
		body := make([]Point, 0, len(s.Body))
		for _, bp := range s.Body {
			body = append(body, Point{X: bp.X, Y: bp.Y})
		}
		snakes = append(snakes, DebugSnakeState{
			ID:     s.ID,
			Health: s.Health,
			Body:   body,
			Alive:  s.Alive,
		})
	}

	food := make([]Point, 0, len(t.Food))
	for _, p := range t.Food {
		food = append(food, Point{X: p.X, Y: p.Y})
	}

	return &DebugGameState{
		Turn:   t.Turn,
		Width:  t.Width,
		Height: t.Height,
		Food:   food,
		Snakes: snakes,
	}
}

// convertJointNodeToDebug converts mcts.JointNode to DebugMCTSNode for the frontend.
func convertJointNodeToDebug(node *mcts.JointNode, cpuct float32) *DebugMCTSNode {
	if node == nil {
		return nil
	}

	q := float32(0)
	if node.VisitCount > 0 {
		q = node.ValueSum / float32(node.VisitCount)
	}

	dn := &DebugMCTSNode{
		VisitCount: node.VisitCount,
		ValueSum:   node.ValueSum,
		Q:          q,
		PriorProb:  1.0,
		UCB:        q,
	}

	if node.State != nil {
		dn.State = gameStateToDebugState(node.State)
	}

	// Convert snake priors
	if len(node.SnakePriors) > 0 {
		dn.SnakePriors = make(map[string][4]float32)
		for sid, priors := range node.SnakePriors {
			dn.SnakePriors[sid] = priors
		}
	}

	// Convert children
	if len(node.Children) > 0 {
		dn.Children = make([]*DebugMCTSNode, 0, len(node.Children))

		// Sort by visit count descending
		type childEntry struct {
			key   string
			child *mcts.JointChild
		}
		entries := make([]childEntry, 0, len(node.Children))
		for k, c := range node.Children {
			entries = append(entries, childEntry{k, c})
		}
		for i := 0; i < len(entries)-1; i++ {
			for j := i + 1; j < len(entries); j++ {
				if entries[j].child.VisitCount > entries[i].child.VisitCount {
					entries[i], entries[j] = entries[j], entries[i]
				}
			}
		}

		sqrtN := float32(1)
		if node.VisitCount > 0 {
			sqrtN = float32(math.Sqrt(float64(node.VisitCount)))
		}

		for _, entry := range entries {
			child := entry.child
			childQ := float32(0)
			if child.VisitCount > 0 {
				childQ = child.ValueSum / float32(child.VisitCount)
			}

			childDebug := &DebugMCTSNode{
				Moves:      child.Action.Moves,
				VisitCount: child.VisitCount,
				ValueSum:   child.ValueSum,
				Q:          childQ,
				PriorProb:  child.PriorProb,
				UCB:        childQ + cpuct*child.PriorProb*sqrtN/(1+float32(child.VisitCount)),
			}

			// Add state if node exists
			if child.Node != nil && child.Node.State != nil {
				childDebug.State = gameStateToDebugState(child.Node.State)
				// Recursively convert children
				if child.Node.IsExpanded && len(child.Node.Children) > 0 {
					childDebug.Children = convertJointNodeToDebug(child.Node, cpuct).Children
				}
			}

			dn.Children = append(dn.Children, childDebug)
		}
	}

	return dn
}

// gameStateToDebugState converts a game.GameState to DebugGameState.
func gameStateToDebugState(s *gamepkg.GameState) *DebugGameState {
	if s == nil {
		return nil
	}

	snakes := make([]DebugSnakeState, 0, len(s.Snakes))
	for _, sn := range s.Snakes {
		body := make([]Point, 0, len(sn.Body))
		for _, bp := range sn.Body {
			body = append(body, Point{X: bp.X, Y: bp.Y})
		}
		snakes = append(snakes, DebugSnakeState{
			ID:     sn.Id,
			Health: sn.Health,
			Body:   body,
			Alive:  sn.Health > 0,
		})
	}

	food := make([]Point, 0, len(s.Food))
	for _, p := range s.Food {
		food = append(food, Point{X: p.X, Y: p.Y})
	}

	return &DebugGameState{
		Turn:   s.Turn,
		Width:  s.Width,
		Height: s.Height,
		Food:   food,
		Snakes: snakes,
	}
}

// convertSimultaneousNodeToDebug converts a simultaneous MCTS tree to debug format.
// The tree structure is the same as alternating, but all snakes in a round share
// the same base state and cached inference.
func convertSimultaneousNodeToDebug(node *mcts.SimultaneousNode, snakeOrder []string, cpuct float32) *DebugMCTSNode {
	if node == nil {
		return nil
	}

	q := float32(0)
	if node.VisitCount > 0 {
		q = node.ValueSum / float32(node.VisitCount)
	}

	dn := &DebugMCTSNode{
		VisitCount: node.VisitCount,
		ValueSum:   node.ValueSum,
		Q:          q,
		PriorProb:  1.0,
		UCB:        q,
		SnakeID:    node.SnakeID,
		SnakeIndex: node.SnakeIndex,
	}

	if node.State != nil {
		dn.State = gameStateToDebugState(node.State)
	}

	// Include the cached priors if available (shows what NN predicted for each snake)
	if node.RoundCache != nil && len(node.RoundCache.Policies) > 0 {
		dn.SnakePriors = node.RoundCache.Policies
	}

	// Convert children
	if len(node.Children) > 0 {
		dn.Children = make([]*DebugMCTSNode, 0, len(node.Children))

		// Sort by visit count descending
		children := make([]*mcts.SimultaneousChild, len(node.Children))
		copy(children, node.Children)
		for i := 0; i < len(children)-1; i++ {
			for j := i + 1; j < len(children); j++ {
				if children[j].VisitCount > children[i].VisitCount {
					children[i], children[j] = children[j], children[i]
				}
			}
		}

		sqrtN := float32(1)
		if node.VisitCount > 0 {
			sqrtN = float32(math.Sqrt(float64(node.VisitCount)))
		}

		// For non-first snakes, calculate aggregated stats for display
		var aggStats *[4]mcts.MoveStats
		var aggTotalVisits int
		if node.SnakeIndex > 0 && node.RoundCache != nil && node.RoundCache.AggregatedStats != nil {
			aggStats = node.RoundCache.AggregatedStats[node.SnakeID]
			if aggStats != nil {
				for i := 0; i < 4; i++ {
					aggTotalVisits += aggStats[i].VisitCount
				}
			}
		}
		sqrtAggN := float32(1)
		if aggTotalVisits > 0 {
			sqrtAggN = float32(math.Sqrt(float64(aggTotalVisits)))
		}

		for _, child := range children {
			if child.Node == nil {
				continue
			}

			childDebug := convertSimultaneousNodeToDebug(child.Node, snakeOrder, cpuct)
			if childDebug == nil {
				continue
			}

			// The move that was made to reach this child (by parent's snake)
			childDebug.Moves = map[string]int{node.SnakeID: child.Move}

			// For non-first snakes, show AGGREGATED stats (what the search actually uses)
			if aggStats != nil {
				stats := &aggStats[child.Move]
				childDebug.VisitCount = stats.VisitCount
				childDebug.ValueSum = stats.ValueSum
				if stats.VisitCount > 0 {
					childDebug.Q = stats.ValueSum / float32(stats.VisitCount)
				} else {
					childDebug.Q = 0
				}
				childDebug.UCB = childDebug.Q + cpuct*child.PriorProb*sqrtAggN/(1+float32(stats.VisitCount))
			} else {
				// First snake: use branch-specific stats
				childDebug.VisitCount = child.VisitCount
				childDebug.ValueSum = child.ValueSum
				if child.VisitCount > 0 {
					childDebug.Q = child.ValueSum / float32(child.VisitCount)
				} else {
					childDebug.Q = 0
				}
				childDebug.UCB = childDebug.Q + cpuct*child.PriorProb*sqrtN/(1+float32(child.VisitCount))
			}
			childDebug.PriorProb = child.PriorProb

			dn.Children = append(dn.Children, childDebug)
		}
	}

	return dn
}
