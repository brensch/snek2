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

	// Use cached games index for fast pagination
	gamesIndex, err := s.dbCache.GetGamesIndex(r.Context())
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	limit := parseIntQuery(r, "limit", 200)
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
