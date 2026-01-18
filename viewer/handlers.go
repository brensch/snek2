package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/brensch/snek2/executor/inference"
	"github.com/brensch/snek2/executor/mcts"
	gamepkg "github.com/brensch/snek2/game"
	"github.com/brensch/snek2/rules"
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
	mux.HandleFunc("/api/mcts", s.handleMCTS)
	mux.HandleFunc("/api/mcts_all", s.handleMCTSAll)
	mux.HandleFunc("/api/simulate", s.handleSimulate)
	mux.HandleFunc("/api/debug_games", s.handleDebugGamesList)
	mux.HandleFunc("/api/debug_games/", s.handleDebugGame)
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

func (s *Server) handleMCTS(w http.ResponseWriter, r *http.Request) {
	withCORS(w, r)
	if r.Method == http.MethodOptions {
		return
	}
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	gameID := strings.TrimSpace(r.URL.Query().Get("game_id"))
	turn := parseIntQuery(r, "turn", -1)
	egoIdx := parseIntQuery(r, "ego_idx", 0)
	sims := parseIntQuery(r, "sims", 100)
	depth := parseIntQuery(r, "depth", 3)
	cpuctStr := strings.TrimSpace(r.URL.Query().Get("cpuct"))
	cpuct := float32(1.0)
	if cpuctStr != "" {
		if f, err := strconv.ParseFloat(cpuctStr, 32); err == nil {
			cpuct = float32(f)
		}
	}

	if gameID == "" || turn < 0 {
		http.Error(w, "missing game_id or turn", http.StatusBadRequest)
		return
	}
	if sims <= 0 {
		sims = 100
	}
	if depth < 0 {
		depth = 0
	}

	p, err := s.getPool()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	db, err := s.dbCache.Get()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()

	t, err := queryTurn(ctx, db, gameID, int32(turn))
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			http.NotFound(w, r)
			return
		}
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	state, egoID, err := turnToGameState(t, egoIdx)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	m := &mcts.MCTS{Config: mcts.Config{Cpuct: cpuct}, Client: p, Rng: rand.New(rand.NewSource(time.Now().UnixNano()))}
	root, maxDepth, err := m.Search(ctx, state, sims)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	bestMove := -1
	bestN := -1
	for i := 0; i < 4; i++ {
		child := root.Children[mcts.Move(i)]
		if child == nil {
			continue
		}
		if child.VisitCount > bestN {
			bestN = child.VisitCount
			bestMove = i
		}
	}

	resp := MCTSResponse{
		GameID:   gameID,
		Turn:     t.Turn,
		EgoIdx:   egoIdx,
		EgoID:    egoID,
		Sims:     sims,
		Cpuct:    cpuct,
		Depth:    depth,
		MaxDepth: maxDepth,
		BestMove: bestMove,
		Root:     buildMCTSNode(gameID, root, depth, cpuct),
	}
	writeJSON(w, resp)
}

func (s *Server) handleMCTSAll(w http.ResponseWriter, r *http.Request) {
	withCORS(w, r)
	if r.Method == http.MethodOptions {
		return
	}
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	gameID := strings.TrimSpace(r.URL.Query().Get("game_id"))
	turn := parseIntQuery(r, "turn", -1)
	sims := parseIntQuery(r, "sims", 100)
	depth := parseIntQuery(r, "depth", 3)
	cpuctStr := strings.TrimSpace(r.URL.Query().Get("cpuct"))
	cpuct := float32(1.0)
	if cpuctStr != "" {
		if f, err := strconv.ParseFloat(cpuctStr, 32); err == nil {
			cpuct = float32(f)
		}
	}

	if gameID == "" || turn < 0 {
		http.Error(w, "missing game_id or turn", http.StatusBadRequest)
		return
	}
	if sims <= 0 {
		sims = 100
	}
	if depth < 0 {
		depth = 0
	}

	p, err := s.getPool()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	db, err := s.dbCache.Get()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 60*time.Second)
	defer cancel()

	t, err := queryTurn(ctx, db, gameID, int32(turn))
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			http.NotFound(w, r)
			return
		}
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	baseState, err := turnToGameStateBase(t)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	out := MCTSAllResponse{GameID: gameID, Turn: t.Turn, Sims: sims, Cpuct: cpuct, Depth: depth, State: t}
	out.Snakes = make([]MCTSSnakeTree, len(t.Snakes))

	var wg sync.WaitGroup
	errCh := make(chan error, 1)
	for idx := range t.Snakes {
		snake := t.Snakes[idx]
		if !snake.Alive || snake.Health <= 0 || len(snake.Body) == 0 {
			out.Snakes[idx] = MCTSSnakeTree{SnakeIdx: idx, SnakeID: snake.ID, BestMove: -1, Root: nil}
			continue
		}

		wg.Add(1)
		go func(snakeIdx int, snakeID string) {
			defer wg.Done()
			local := baseState.Clone()
			local.YouId = snakeID

			m := &mcts.MCTS{Config: mcts.Config{Cpuct: cpuct}, Client: p, Rng: rand.New(rand.NewSource(time.Now().UnixNano()))}
			root, _, err := m.Search(ctx, local, sims)
			if err != nil {
				select {
				case errCh <- err:
				default:
				}
				return
			}

			bestMove := -1
			bestN := -1
			for i := 0; i < 4; i++ {
				child := root.Children[mcts.Move(i)]
				if child == nil {
					continue
				}
				if child.VisitCount > bestN {
					bestN = child.VisitCount
					bestMove = i
				}
			}
			if bestMove < 0 {
				bestMove = 0
			}

			out.Snakes[snakeIdx] = MCTSSnakeTree{
				SnakeIdx: snakeIdx,
				SnakeID:  snakeID,
				BestMove: bestMove,
				Root:     buildMCTSNode(gameID, root, depth, cpuct),
			}
		}(idx, snake.ID)
	}
	wg.Wait()

	select {
	case err := <-errCh:
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	default:
	}

	writeJSON(w, out)
}

func (s *Server) handleSimulate(w http.ResponseWriter, r *http.Request) {
	withCORS(w, r)
	if r.Method == http.MethodOptions {
		return
	}
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req SimulateRequest
	dec := json.NewDecoder(r.Body)
	if err := dec.Decode(&req); err != nil {
		http.Error(w, "bad json", http.StatusBadRequest)
		return
	}
	st, err := turnToGameStateBase(req.State)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if req.Moves == nil {
		req.Moves = map[string]int{}
	}

	next := rules.NextStateSimultaneous(st, req.Moves)
	writeJSON(w, gameStateToTurn(req.State.GameID, next))
}

// turnToGameState converts a Turn to a GameState with the specified ego snake.
func turnToGameState(t Turn, egoIdx int) (*gamepkg.GameState, string, error) {
	if egoIdx < 0 || egoIdx >= len(t.Snakes) {
		return nil, "", fmt.Errorf("ego_idx out of range")
	}
	if t.Width <= 0 || t.Height <= 0 {
		return nil, "", fmt.Errorf("invalid board size")
	}

	food := make([]gamepkg.Point, 0, len(t.Food))
	for _, p := range t.Food {
		food = append(food, gamepkg.Point{X: p.X, Y: p.Y})
	}

	snakes := make([]gamepkg.Snake, 0, len(t.Snakes))
	for _, s := range t.Snakes {
		body := make([]gamepkg.Point, 0, len(s.Body))
		for _, bp := range s.Body {
			body = append(body, gamepkg.Point{X: bp.X, Y: bp.Y})
		}
		snakes = append(snakes, gamepkg.Snake{Id: s.ID, Health: s.Health, Body: body})
	}

	egoID := t.Snakes[egoIdx].ID
	st := &gamepkg.GameState{
		Width:  t.Width,
		Height: t.Height,
		Snakes: snakes,
		Food:   food,
		YouId:  egoID,
		Turn:   t.Turn,
	}
	return st, egoID, nil
}

// turnToGameStateBase converts a Turn to a GameState with the first snake as ego.
func turnToGameStateBase(t Turn) (*gamepkg.GameState, error) {
	if t.Width <= 0 || t.Height <= 0 {
		return nil, fmt.Errorf("invalid board size")
	}
	if len(t.Snakes) == 0 {
		return nil, fmt.Errorf("no snakes")
	}

	food := make([]gamepkg.Point, 0, len(t.Food))
	for _, p := range t.Food {
		food = append(food, gamepkg.Point{X: p.X, Y: p.Y})
	}

	snakes := make([]gamepkg.Snake, 0, len(t.Snakes))
	for _, s := range t.Snakes {
		body := make([]gamepkg.Point, 0, len(s.Body))
		for _, bp := range s.Body {
			body = append(body, gamepkg.Point{X: bp.X, Y: bp.Y})
		}
		snakes = append(snakes, gamepkg.Snake{Id: s.ID, Health: s.Health, Body: body})
	}

	st := &gamepkg.GameState{
		Width:  t.Width,
		Height: t.Height,
		Snakes: snakes,
		Food:   food,
		YouId:  t.Snakes[0].ID,
		Turn:   t.Turn,
	}
	return st, nil
}

// gameStateToTurn converts a GameState back to a Turn.
func gameStateToTurn(gameID string, s *gamepkg.GameState) Turn {
	if s == nil {
		return Turn{}
	}

	out := Turn{
		GameID:  gameID,
		Turn:    s.Turn,
		Width:   s.Width,
		Height:  s.Height,
		Source:  "mcts",
		Food:    make([]Point, 0, len(s.Food)),
		Hazards: nil,
		Snakes:  make([]Snake, 0, len(s.Snakes)),
	}
	for _, p := range s.Food {
		out.Food = append(out.Food, Point{X: p.X, Y: p.Y})
	}
	for _, sn := range s.Snakes {
		body := make([]Point, 0, len(sn.Body))
		for _, bp := range sn.Body {
			body = append(body, Point{X: bp.X, Y: bp.Y})
		}
		out.Snakes = append(out.Snakes, Snake{
			ID:     sn.Id,
			Alive:  sn.Health > 0,
			Health: sn.Health,
			Body:   body,
			Policy: -1,
			Value:  0,
		})
	}
	return out
}

// buildMCTSNode recursively builds the MCTS node tree for JSON output.
func buildMCTSNode(gameID string, n *mcts.Node, depth int, cpuct float32) *MCTSNode {
	if n == nil {
		return nil
	}

	avg := float32(0)
	if n.VisitCount > 0 {
		avg = n.ValueSum / float32(n.VisitCount)
	}

	out := &MCTSNode{VisitCount: n.VisitCount, Value: avg}
	if n.State != nil {
		t := gameStateToTurn(gameID, n.State)
		out.State = &t
	}
	sqrtSumN := float32(math.Sqrt(float64(n.VisitCount)))

	for i := 0; i < 4; i++ {
		child := n.Children[mcts.Move(i)]
		mv := MCTSMove{Move: i}
		if child != nil {
			mv.Exists = true
			mv.N = child.VisitCount
			if child.VisitCount > 0 {
				mv.Q = child.ValueSum / float32(child.VisitCount)
			}
			mv.P = child.PriorProb
			mv.UCB = mv.Q + cpuct*mv.P*sqrtSumN/(1+float32(child.VisitCount))
			if depth > 0 {
				mv.Child = buildMCTSNode(gameID, child, depth-1, cpuct)
			}
		}
		out.Moves[i] = mv
	}

	return out
}

// handleDebugGamesList returns a list of available debug games.
func (s *Server) handleDebugGamesList(w http.ResponseWriter, r *http.Request) {
	withCORS(w, r)
	if r.Method == http.MethodOptions {
		return
	}
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	games, err := listDebugGames()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	writeJSON(w, games)
}

// handleDebugGame returns the full data for a specific debug game.
func (s *Server) handleDebugGame(w http.ResponseWriter, r *http.Request) {
	withCORS(w, r)
	if r.Method == http.MethodOptions {
		return
	}
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// /api/debug_games/{game_id}
	rest := strings.TrimPrefix(r.URL.Path, "/api/debug_games/")
	gameID, err := url.PathUnescape(rest)
	if err != nil || gameID == "" {
		http.Error(w, "bad game id", http.StatusBadRequest)
		return
	}

	game, err := loadDebugGame(gameID)
	if err != nil {
		if os.IsNotExist(err) {
			http.NotFound(w, r)
			return
		}
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	writeJSON(w, game)
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

	// Run alternating MCTS - each snake takes turns in the tree
	mctsConfig := mcts.Config{Cpuct: 1.0}
	root, snakeOrder, err := mcts.AlternatingSearch(ctx, pool, mctsConfig, gameState, req.Sims)
	if err != nil {
		http.Error(w, "MCTS search failed: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// Convert to debug format
	debugTree := convertAlternatingNodeToDebug(root, snakeOrder, mctsConfig.Cpuct)
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

// convertAlternatingNodeToDebug converts an alternating MCTS tree to debug format.
// Each node represents ONE snake's move (not all snakes together).
func convertAlternatingNodeToDebug(node *mcts.AlternatingNode, snakeOrder []string, cpuct float32) *DebugMCTSNode {
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

	// Set the move for this node (single snake's move, not joint)
	// Don't set moves for the root node (which has State at the top level)
	// Root node is identified by having State AND being the initial node
	// Child nodes get their moves set when recursing

	if node.State != nil {
		dn.State = gameStateToDebugState(node.State)
	}

	// Convert children - recursively convert each child node
	if len(node.Children) > 0 {
		dn.Children = make([]*DebugMCTSNode, 0, len(node.Children))

		// Sort by visit count descending
		children := make([]*mcts.AlternatingChild, len(node.Children))
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

		for _, child := range children {
			if child.Node == nil {
				continue
			}

			// Recursively convert the child node first
			childDebug := convertAlternatingNodeToDebug(child.Node, snakeOrder, cpuct)
			if childDebug == nil {
				continue
			}

			// Set the move that was made to reach this child
			// The move is on the edge (child.Move), and it was made by the PARENT's snake
			childDebug.Moves = map[string]int{node.SnakeID: child.Move}

			// Override stats with the edge stats (visit count, value sum from parent's perspective)
			childDebug.VisitCount = child.VisitCount
			childDebug.ValueSum = child.ValueSum
			childDebug.PriorProb = child.PriorProb

			// Recalculate Q and UCB from edge stats
			if child.VisitCount > 0 {
				childDebug.Q = child.ValueSum / float32(child.VisitCount)
			} else {
				childDebug.Q = 0
			}
			childDebug.UCB = childDebug.Q + cpuct*child.PriorProb*sqrtN/(1+float32(child.VisitCount))

			dn.Children = append(dn.Children, childDebug)
		}
	}

	return dn
}
