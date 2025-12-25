package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io/fs"
	"log"
	"math"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/brensch/snek2/executor/inference"
	"github.com/brensch/snek2/executor/mcts"
	gamepkg "github.com/brensch/snek2/game"
	"github.com/brensch/snek2/rules"

	_ "github.com/duckdb/duckdb-go/v2"
)

type GameSummary struct {
	GameID string `json:"game_id"`
	// StartedNs is parsed from game_id for selfplay games with format: selfplay_<unix_nano>_<worker>.
	// Nil for game IDs that do not match that format.
	StartedNs  *int64 `json:"started_ns"`
	MinTurn    int32  `json:"min_turn"`
	MaxTurn    int32  `json:"max_turn"`
	TurnCount  int32  `json:"turn_count"`
	Width      int32  `json:"width"`
	Height     int32  `json:"height"`
	Source     string `json:"source"`
	SourceFile string `json:"file"`
	Results    string `json:"results"`
}

type GamesResponse struct {
	Total int64         `json:"total"`
	Games []GameSummary `json:"games"`
}

type StatsPoint struct {
	TNs int64 `json:"t_ns"`

	SelfplayGames      int64 `json:"selfplay_games"`
	SelfplayTotalTurns int64 `json:"selfplay_total_turns"`
	SelfplayWins       int64 `json:"selfplay_wins"`
	SelfplayDraws      int64 `json:"selfplay_draws"`

	ScrapedGames      int64 `json:"scraped_games"`
	ScrapedTotalTurns int64 `json:"scraped_total_turns"`
	ScrapedWins       int64 `json:"scraped_wins"`
	ScrapedDraws      int64 `json:"scraped_draws"`
}

type StatsResponse struct {
	FromNs   int64        `json:"from_ns"`
	ToNs     int64        `json:"to_ns"`
	BucketNs int64        `json:"bucket_ns"`
	Points   []StatsPoint `json:"points"`
}

type Point struct {
	X int32 `json:"x"`
	Y int32 `json:"y"`
}

type Snake struct {
	ID          string    `json:"id"`
	Alive       bool      `json:"alive"`
	Health      int32     `json:"health"`
	Body        []Point   `json:"body"`
	Policy      int32     `json:"policy"`
	PolicyProbs []float32 `json:"policy_probs,omitempty"`
	Value       float32   `json:"value"`
}

type Turn struct {
	GameID  string  `json:"game_id"`
	Turn    int32   `json:"turn"`
	Width   int32   `json:"width"`
	Height  int32   `json:"height"`
	Food    []Point `json:"food"`
	Hazards []Point `json:"hazards"`
	Snakes  []Snake `json:"snakes"`
	Source  string  `json:"source"`
}

type MCTSMove struct {
	Move   int       `json:"move"`
	Exists bool      `json:"exists"`
	N      int       `json:"n"`
	Q      float32   `json:"q"`
	P      float32   `json:"p"`
	UCB    float32   `json:"ucb"`
	Child  *MCTSNode `json:"child,omitempty"`
}

type MCTSNode struct {
	VisitCount int         `json:"visit_count"`
	Value      float32     `json:"value"`
	State      *Turn       `json:"state,omitempty"`
	Moves      [4]MCTSMove `json:"moves"`
}

type MCTSResponse struct {
	GameID   string    `json:"game_id"`
	Turn     int32     `json:"turn"`
	EgoIdx   int       `json:"ego_idx"`
	EgoID    string    `json:"ego_id"`
	Sims     int       `json:"sims"`
	Cpuct    float32   `json:"cpuct"`
	Depth    int       `json:"depth"`
	MaxDepth int       `json:"max_depth"`
	BestMove int       `json:"best_move"`
	Root     *MCTSNode `json:"root"`
}

type MCTSSnakeTree struct {
	SnakeIdx int       `json:"snake_idx"`
	SnakeID  string    `json:"snake_id"`
	BestMove int       `json:"best_move"`
	Root     *MCTSNode `json:"root"`
}

// MCTSAllResponse emulates the selfplay generation logic:
// run an independent MCTS for each living snake (each with its own YouId perspective)
// on the same position.
type MCTSAllResponse struct {
	GameID string          `json:"game_id"`
	Turn   int32           `json:"turn"`
	Sims   int             `json:"sims"`
	Cpuct  float32         `json:"cpuct"`
	Depth  int             `json:"depth"`
	State  Turn            `json:"state"`
	Snakes []MCTSSnakeTree `json:"snakes"`
}

type SimulateRequest struct {
	State Turn           `json:"state"`
	Moves map[string]int `json:"moves"`
}

func main() {
	fs := flag.NewFlagSet(os.Args[0], flag.ContinueOnError)
	fs.SetOutput(os.Stderr)

	listen := fs.String("listen", "127.0.0.1:8080", "HTTP listen address")
	dataDir := fs.String("data-dir", "", "Directory containing archive parquet shards (archive_turn_v1) [deprecated: prefer -data-dirs]")
	dataDirs := fs.String("data-dirs", strings.Join(defaultDataDirs(), ","), "Comma-separated list of directories containing archive parquet shards (archive_turn_v1)")
	staticDir := fs.String("static-dir", "", "Optional directory to serve as SPA static (e.g. viewer/web/dist)")
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

	mux := http.NewServeMux()
	mux.HandleFunc("/api/games", func(w http.ResponseWriter, r *http.Request) {
		withCORS(w, r)
		if r.Method == http.MethodOptions {
			return
		}
		if r.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		db, err := openDuckDBForRoots(roots)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		defer db.Close()

		limit := parseIntQuery(r, "limit", 200)
		offset := parseIntQuery(r, "offset", 0)
		sortKey := strings.TrimSpace(r.URL.Query().Get("sort"))
		sortDir := strings.TrimSpace(r.URL.Query().Get("dir"))
		total, err := queryGamesTotal(r.Context(), db)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		games, err := queryGames(r.Context(), db, roots, limit, offset, sortKey, sortDir)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		writeJSON(w, GamesResponse{Total: total, Games: games})
	})

	mux.HandleFunc("/api/games/", func(w http.ResponseWriter, r *http.Request) {
		withCORS(w, r)
		if r.Method == http.MethodOptions {
			return
		}
		if r.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		db, err := openDuckDBForRoots(roots)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		defer db.Close()

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
	})

	mux.HandleFunc("/api/stats", func(w http.ResponseWriter, r *http.Request) {
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

		db, err := openDuckDBForRoots(roots)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		defer db.Close()

		points, err := queryStats(r.Context(), db, fromNs, toNs, bucketNs)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		writeJSON(w, StatsResponse{FromNs: fromNs, ToNs: toNs, BucketNs: bucketNs, Points: points})
	})

	mux.HandleFunc("/api/mcts", func(w http.ResponseWriter, r *http.Request) {
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
		sims := parseIntQuery(r, "sims", 800)
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
			sims = 800
		}
		if depth < 0 {
			depth = 0
		}

		p, err := getPool()
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		db, err := openDuckDBForRoots(roots)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		defer db.Close()

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

		m := &mcts.MCTS{Config: mcts.Config{Cpuct: cpuct}, Client: p}
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
	})

	mux.HandleFunc("/api/mcts_all", func(w http.ResponseWriter, r *http.Request) {
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
		sims := parseIntQuery(r, "sims", 800)
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
			sims = 800
		}
		if depth < 0 {
			depth = 0
		}

		p, err := getPool()
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		db, err := openDuckDBForRoots(roots)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		defer db.Close()

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
			s := t.Snakes[idx]
			if !s.Alive || s.Health <= 0 || len(s.Body) == 0 {
				out.Snakes[idx] = MCTSSnakeTree{SnakeIdx: idx, SnakeID: s.ID, BestMove: -1, Root: nil}
				continue
			}

			wg.Add(1)
			go func(snakeIdx int, snakeID string) {
				defer wg.Done()
				local := baseState.Clone()
				local.YouId = snakeID

				m := &mcts.MCTS{Config: mcts.Config{Cpuct: cpuct}, Client: p}
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
			}(idx, s.ID)
		}
		wg.Wait()

		select {
		case err := <-errCh:
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		default:
		}

		writeJSON(w, out)
	})

	mux.HandleFunc("/api/simulate", func(w http.ResponseWriter, r *http.Request) {
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
	})

	if strings.TrimSpace(*staticDir) != "" {
		spa := spaHandler{staticPath: *staticDir, indexPath: filepath.Join(*staticDir, "index.html")}
		mux.Handle("/", spa)
	}

	srv := &http.Server{
		Addr:              *listen,
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
	}
	log.Printf("Viewer API listening on http://%s", *listen)
	if strings.TrimSpace(*staticDir) != "" {
		log.Printf("Serving SPA from %s", *staticDir)
	}
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

type spaHandler struct {
	staticPath string
	indexPath  string
}

func (h spaHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// Serve exact static asset if exists; otherwise serve index.html for client-side routing.
	path := filepath.Clean(r.URL.Path)
	if path == "/" {
		http.ServeFile(w, r, h.indexPath)
		return
	}
	candidate := filepath.Join(h.staticPath, strings.TrimPrefix(path, "/"))
	if fi, err := os.Stat(candidate); err == nil && !fi.IsDir() {
		http.ServeFile(w, r, candidate)
		return
	}
	// Fallback to SPA.
	http.ServeFile(w, r, h.indexPath)
}

func withCORS(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	enc := json.NewEncoder(w)
	_ = enc.Encode(v)
}

func parseIntQuery(r *http.Request, key string, def int) int {
	v := strings.TrimSpace(r.URL.Query().Get(key))
	if v == "" {
		return def
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return def
	}
	if n < 0 {
		return def
	}
	return n
}

func parseInt64Query(r *http.Request, key string, def int64) int64 {
	v := strings.TrimSpace(r.URL.Query().Get(key))
	if v == "" {
		return def
	}
	n, err := strconv.ParseInt(v, 10, 64)
	if err != nil {
		return def
	}
	return n
}

func findParquetFiles(root string) ([]string, error) {
	root = strings.TrimSpace(root)
	if root == "" {
		return nil, nil
	}
	var files []string
	walkErr := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		name := d.Name()
		if d.IsDir() {
			if name == "tmp" {
				return filepath.SkipDir
			}
			return nil
		}
		if strings.HasSuffix(strings.ToLower(name), ".parquet") {
			files = append(files, path)
		}
		return nil
	})
	if walkErr != nil {
		if os.IsNotExist(walkErr) {
			return nil, nil
		}
		return nil, walkErr
	}
	return files, nil
}

func findParquetFilesMulti(roots []string) ([]string, error) {
	seen := make(map[string]bool, 1024)
	out := make([]string, 0, 1024)
	for _, r := range roots {
		files, err := findParquetFiles(r)
		if err != nil {
			return nil, err
		}
		for _, f := range files {
			if seen[f] {
				continue
			}
			seen[f] = true
			out = append(out, f)
		}
	}
	return out, nil
}

func openDuckDBForRoots(roots []string) (*sql.DB, error) {
	parquetFiles, err := findParquetFilesMulti(roots)
	if err != nil {
		return nil, err
	}
	return openDuckDB(parquetFiles)
}

func openDuckDB(parquetFiles []string) (*sql.DB, error) {
	db, err := sql.Open("duckdb", ":memory:")
	if err != nil {
		return nil, err
	}
	// Basic pragmas; ignore errors for compatibility across versions.
	_, _ = db.Exec("PRAGMA threads=4")
	// Disable DuckDB's object cache so API responses reflect on-disk changes.
	_, _ = db.Exec("PRAGMA enable_object_cache=false")

	// Create a view over all parquet shards.
	if len(parquetFiles) == 0 {
		_, err := db.Exec(`CREATE OR REPLACE VIEW turns AS
			SELECT * FROM (
				SELECT
					NULL::VARCHAR AS game_id,
					NULL::INTEGER AS turn,
					NULL::INTEGER AS width,
					NULL::INTEGER AS height,
					NULL::INTEGER[] AS food_x,
					NULL::INTEGER[] AS food_y,
					NULL::INTEGER[] AS hazard_x,
					NULL::INTEGER[] AS hazard_y,
					NULL::STRUCT(
						id VARCHAR,
						alive BOOLEAN,
						health INTEGER,
						body_x INTEGER[],
						body_y INTEGER[],
						policy INTEGER,
						policy_probs REAL[],
						value REAL
					)[] AS snakes,
					NULL::VARCHAR AS filename,
					NULL::VARCHAR AS source
			) WHERE 1=0`)
		if err != nil {
			_ = db.Close()
			return nil, err
		}
		return db, nil
	}

	arr := make([]string, 0, len(parquetFiles))
	for _, p := range parquetFiles {
		arr = append(arr, "'"+escapeSQLString(p)+"'")
	}
	// filename=true adds a 'filename' column so we can show provenance in the UI.
	sqlText := "CREATE OR REPLACE VIEW turns AS SELECT * FROM read_parquet([" + strings.Join(arr, ",") + "], filename=true)"
	if _, err := db.Exec(sqlText); err != nil {
		_ = db.Close()
		return nil, err
	}
	return db, nil
}

func escapeSQLString(s string) string {
	return strings.ReplaceAll(s, "'", "''")
}

func queryGamesTotal(ctx context.Context, db *sql.DB) (int64, error) {
	var total int64
	// DuckDB supports COUNT(DISTINCT ...).
	if err := db.QueryRowContext(ctx, `SELECT COUNT(DISTINCT game_id) FROM turns`).Scan(&total); err != nil {
		return 0, err
	}
	return total, nil
}

func normalizeSort(sortKey string, sortDir string) (string, string) {
	sk := strings.ToLower(strings.TrimSpace(sortKey))
	sd := strings.ToLower(strings.TrimSpace(sortDir))
	if sd != "asc" && sd != "desc" {
		sd = "desc"
	}
	// Map user-facing keys to SQL expressions / aliases. Must be safe (no user input concatenated).
	switch sk {
	case "time", "started", "started_ns":
		sk = "started_ns"
	case "id", "game", "game_id":
		sk = "game_id"
	case "turns", "turn_count":
		sk = "turn_count"
	case "width":
		sk = "width"
	case "height":
		sk = "height"
	case "source":
		sk = "source"
	case "file", "filename":
		sk = "file"
	default:
		sk = "started_ns"
		sd = "desc"
	}
	return sk, sd
}

func makeRelativeToRoots(filename string, roots []string) string {
	fn := strings.TrimSpace(filename)
	if fn == "" {
		return ""
	}
	best := fn
	bestLen := len(best)
	for _, r := range roots {
		root := strings.TrimSpace(r)
		if root == "" {
			continue
		}
		rel, err := filepath.Rel(root, fn)
		if err != nil {
			continue
		}
		// Ignore paths that escape the root.
		if strings.HasPrefix(rel, "..") {
			continue
		}
		cand := filepath.ToSlash(filepath.Join(root, rel))
		if len(cand) < bestLen {
			best = cand
			bestLen = len(cand)
		}
	}
	return best
}

func queryGames(ctx context.Context, db *sql.DB, roots []string, limit int, offset int, sortKey string, sortDir string) ([]GameSummary, error) {
	sk, sd := normalizeSort(sortKey, sortDir)
	orderExpr := "started_ns"
	switch sk {
	case "started_ns":
		orderExpr = "started_ns"
	case "game_id":
		orderExpr = "game_id"
	case "turn_count":
		orderExpr = "turn_count"
	case "width":
		orderExpr = "width"
	case "height":
		orderExpr = "height"
	case "source":
		orderExpr = "source"
	case "file":
		orderExpr = "file"
	}

	orderClause := " ORDER BY " + orderExpr + " " + strings.ToUpper(sd)
	if orderExpr == "started_ns" {
		orderClause += " NULLS LAST"
	}
	// Stable tie-breakers.
	if orderExpr != "started_ns" {
		orderClause += ", started_ns DESC NULLS LAST"
	}
	orderClause += ", game_id DESC"

	query :=
		`SELECT
			game_id,
			CASE
				WHEN starts_with(game_id, 'selfplay_') THEN try_cast(regexp_extract(game_id, '^selfplay_([0-9]+)_', 1) AS BIGINT)
				ELSE NULL
			END AS started_ns,
			MIN(turn)::INTEGER AS min_turn,
			MAX(turn)::INTEGER AS max_turn,
			(MAX(turn) - MIN(turn) + 1)::INTEGER AS turn_count,
			MIN(width)::INTEGER AS width,
			MIN(height)::INTEGER AS height,
			MIN(source)::VARCHAR AS source,
			MIN(filename)::VARCHAR AS file
		FROM turns
		GROUP BY game_id` +
			orderClause +
			` LIMIT ? OFFSET ?`

	rows, err := db.QueryContext(ctx,
		query, limit, offset)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := make([]GameSummary, 0, limit)
	for rows.Next() {
		var g GameSummary
		var file string
		if err := rows.Scan(&g.GameID, &g.StartedNs, &g.MinTurn, &g.MaxTurn, &g.TurnCount, &g.Width, &g.Height, &g.Source, &file); err != nil {
			return nil, err
		}
		g.SourceFile = makeRelativeToRoots(file, roots)
		out = append(out, g)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	if len(out) == 0 {
		return out, nil
	}

	gameIDs := make([]string, 0, len(out))
	for _, g := range out {
		gameIDs = append(gameIDs, g.GameID)
	}
	results, err := queryGameResults(ctx, db, gameIDs)
	if err != nil {
		return nil, err
	}
	for i := range out {
		out[i].Results = results[out[i].GameID]
	}
	return out, nil
}

func queryGameResults(ctx context.Context, db *sql.DB, gameIDs []string) (map[string]string, error) {
	if len(gameIDs) == 0 {
		return map[string]string{}, nil
	}
	args := make([]any, 0, len(gameIDs))
	placeholders := make([]string, 0, len(gameIDs))
	for _, id := range gameIDs {
		placeholders = append(placeholders, "?")
		args = append(args, id)
	}

	query := `WITH last_turn AS (
		SELECT game_id, snakes
		FROM (
			SELECT
				game_id,
				snakes,
				row_number() OVER (PARTITION BY game_id ORDER BY turn DESC) AS rn
			FROM turns
			WHERE game_id IN (` + strings.Join(placeholders, ",") + `)
		)
		WHERE rn = 1
	)
	SELECT game_id, snakes FROM last_turn`

	rows, err := db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := make(map[string]string, len(gameIDs))
	for rows.Next() {
		var gameID string
		var snakesAny any
		if err := rows.Scan(&gameID, &snakesAny); err != nil {
			return nil, err
		}
		snakes := asSnakes(snakesAny)
		out[gameID] = formatSnakeResults(snakes)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return out, nil
}

func formatSnakeResults(snakes []Snake) string {
	if len(snakes) == 0 {
		return ""
	}
	parts := make([]string, 0, len(snakes))
	for i, s := range snakes {
		letter := string(rune('A' + (i % 26)))
		val := strconv.FormatFloat(float64(s.Value), 'f', -1, 32)
		parts = append(parts, letter+":"+val)
	}
	return strings.Join(parts, " ")
}

func queryStats(ctx context.Context, db *sql.DB, fromNs int64, toNs int64, bucketNs int64) ([]StatsPoint, error) {
	query := `WITH games AS (
		SELECT
			game_id,
			CASE WHEN starts_with(game_id, 'selfplay_') THEN 'selfplay' ELSE 'scraped' END AS kind,
			COALESCE(
				try_cast(regexp_extract(game_id, '^selfplay_([0-9]+)_', 1) AS BIGINT),
				try_cast(regexp_extract(MIN(filename), 'batch_([0-9]+)\\.parquet', 1) AS BIGINT),
				try_cast(regexp_extract(MIN(filename), 'batch_([0-9]+)', 1) AS BIGINT)
			) AS ts_ns,
			MIN(turn)::BIGINT AS min_turn,
			MAX(turn)::BIGINT AS max_turn,
			(MAX(turn) - MIN(turn) + 1)::BIGINT AS turn_count
		FROM turns
		GROUP BY game_id
	),
	filtered AS (
		SELECT *
		FROM games
		WHERE ts_ns IS NOT NULL
			AND ts_ns >= ?
			AND ts_ns <= ?
	),
	last_turn AS (
		SELECT t.game_id, t.snakes
		FROM turns t
		JOIN filtered g ON g.game_id = t.game_id AND t.turn = g.max_turn
	),
	alive_counts AS (
		SELECT game_id,
			SUM(CASE WHEN s.alive THEN 1 ELSE 0 END)::BIGINT AS alive_count
		FROM last_turn, UNNEST(snakes) AS u(s)
		GROUP BY game_id
	),
	joined AS (
		SELECT
			f.game_id,
			f.kind,
			f.ts_ns,
			f.turn_count,
			COALESCE(a.alive_count, 0)::BIGINT AS alive_count,
			(? + floor((f.ts_ns - ?)::DOUBLE / ?::DOUBLE) * ?)::BIGINT AS bucket_start_ns
		FROM filtered f
		LEFT JOIN alive_counts a ON a.game_id = f.game_id
	)
	SELECT
		bucket_start_ns,
		SUM(CASE WHEN kind = 'selfplay' THEN 1 ELSE 0 END)::BIGINT AS selfplay_games,
		SUM(CASE WHEN kind = 'selfplay' THEN turn_count ELSE 0 END)::BIGINT AS selfplay_total_turns,
		SUM(CASE WHEN kind = 'selfplay' AND alive_count = 1 THEN 1 ELSE 0 END)::BIGINT AS selfplay_wins,
		SUM(CASE WHEN kind = 'selfplay' AND alive_count <> 1 THEN 1 ELSE 0 END)::BIGINT AS selfplay_draws,
		SUM(CASE WHEN kind = 'scraped' THEN 1 ELSE 0 END)::BIGINT AS scraped_games,
		SUM(CASE WHEN kind = 'scraped' THEN turn_count ELSE 0 END)::BIGINT AS scraped_total_turns,
		SUM(CASE WHEN kind = 'scraped' AND alive_count = 1 THEN 1 ELSE 0 END)::BIGINT AS scraped_wins,
		SUM(CASE WHEN kind = 'scraped' AND alive_count <> 1 THEN 1 ELSE 0 END)::BIGINT AS scraped_draws
	FROM joined
	GROUP BY bucket_start_ns
	ORDER BY bucket_start_ns ASC`

	rows, err := db.QueryContext(ctx, query, fromNs, toNs, fromNs, fromNs, bucketNs, bucketNs)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	points := make([]StatsPoint, 0, 1024)
	for rows.Next() {
		var p StatsPoint
		if err := rows.Scan(
			&p.TNs,
			&p.SelfplayGames,
			&p.SelfplayTotalTurns,
			&p.SelfplayWins,
			&p.SelfplayDraws,
			&p.ScrapedGames,
			&p.ScrapedTotalTurns,
			&p.ScrapedWins,
			&p.ScrapedDraws,
		); err != nil {
			return nil, err
		}
		points = append(points, p)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return points, nil
}

func queryTurns(ctx context.Context, db *sql.DB, gameID string) ([]Turn, error) {
	rows, err := db.QueryContext(ctx,
		`SELECT game_id, turn::INTEGER, width::INTEGER, height::INTEGER, food_x, food_y, hazard_x, hazard_y, snakes, source
		 FROM turns
		 WHERE game_id = ?
		 ORDER BY turn ASC`, gameID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	turns := make([]Turn, 0, 256)
	for rows.Next() {
		var t Turn
		var foodXAny any
		var foodYAny any
		var hazXAny any
		var hazYAny any
		var snakesAny any
		if err := rows.Scan(&t.GameID, &t.Turn, &t.Width, &t.Height, &foodXAny, &foodYAny, &hazXAny, &hazYAny, &snakesAny, &t.Source); err != nil {
			return nil, err
		}
		t.Food = zipPoints(asInt32Slice(foodXAny), asInt32Slice(foodYAny))
		t.Hazards = zipPoints(asInt32Slice(hazXAny), asInt32Slice(hazYAny))
		t.Snakes = asSnakes(snakesAny)
		turns = append(turns, t)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return turns, nil
}

func queryTurn(ctx context.Context, db *sql.DB, gameID string, turn int32) (Turn, error) {
	row := db.QueryRowContext(ctx,
		`SELECT game_id, turn::INTEGER, width::INTEGER, height::INTEGER, food_x, food_y, hazard_x, hazard_y, snakes, source
		 FROM turns
		 WHERE game_id = ? AND turn = ?
		 LIMIT 1`, gameID, turn)

	var t Turn
	var foodXAny any
	var foodYAny any
	var hazXAny any
	var hazYAny any
	var snakesAny any
	if err := row.Scan(&t.GameID, &t.Turn, &t.Width, &t.Height, &foodXAny, &foodYAny, &hazXAny, &hazYAny, &snakesAny, &t.Source); err != nil {
		return Turn{}, err
	}
	t.Food = zipPoints(asInt32Slice(foodXAny), asInt32Slice(foodYAny))
	t.Hazards = zipPoints(asInt32Slice(hazXAny), asInt32Slice(hazYAny))
	t.Snakes = asSnakes(snakesAny)
	return t, nil
}

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

func zipPoints(xs, ys []int32) []Point {
	n := len(xs)
	if len(ys) < n {
		n = len(ys)
	}
	out := make([]Point, 0, n)
	for i := 0; i < n; i++ {
		out = append(out, Point{X: xs[i], Y: ys[i]})
	}
	return out
}

func asInt32Slice(v any) []int32 {
	if v == nil {
		return nil
	}
	switch vv := v.(type) {
	case []int32:
		return vv
	case []int64:
		out := make([]int32, 0, len(vv))
		for _, x := range vv {
			out = append(out, int32(x))
		}
		return out
	case []any:
		out := make([]int32, 0, len(vv))
		for _, x := range vv {
			out = append(out, int32(asInt64(x)))
		}
		return out
	default:
		return nil
	}
}

func asInt64(v any) int64 {
	switch t := v.(type) {
	case int64:
		return t
	case int32:
		return int64(t)
	case int:
		return int64(t)
	case uint64:
		return int64(t)
	case float64:
		return int64(t)
	default:
		return 0
	}
}

func asFloat32(v any) float32 {
	switch t := v.(type) {
	case float32:
		return t
	case float64:
		return float32(t)
	case int64:
		return float32(t)
	case int32:
		return float32(t)
	default:
		return 0
	}
}

func asFloat32Slice(v any) []float32 {
	if v == nil {
		return nil
	}
	switch vv := v.(type) {
	case []float32:
		return vv
	case []float64:
		out := make([]float32, 0, len(vv))
		for _, x := range vv {
			out = append(out, float32(x))
		}
		return out
	case []any:
		out := make([]float32, 0, len(vv))
		for _, x := range vv {
			out = append(out, asFloat32(x))
		}
		return out
	default:
		return nil
	}
}

func asBool(v any) bool {
	switch t := v.(type) {
	case bool:
		return t
	case int64:
		return t != 0
	case int32:
		return t != 0
	default:
		return false
	}
}

func asString(v any) string {
	switch t := v.(type) {
	case string:
		return t
	case []byte:
		return string(t)
	default:
		return ""
	}
}

func asSnakes(v any) []Snake {
	if v == nil {
		return nil
	}
	list, ok := v.([]any)
	if !ok {
		if l2, ok2 := v.([]interface{}); ok2 {
			list = make([]any, len(l2))
			copy(list, l2)
		} else {
			return nil
		}
	}

	snakes := make([]Snake, 0, len(list))
	for _, it := range list {
		m, ok := it.(map[string]any)
		if !ok {
			if m2, ok2 := it.(map[string]interface{}); ok2 {
				m = make(map[string]any, len(m2))
				for k, v := range m2 {
					m[k] = v
				}
			} else {
				continue
			}
		}

		bodyX := asInt32Slice(m["body_x"])
		bodyY := asInt32Slice(m["body_y"])
		s := Snake{
			ID:          asString(m["id"]),
			Alive:       asBool(m["alive"]),
			Health:      int32(asInt64(m["health"])),
			Policy:      int32(asInt64(m["policy"])),
			PolicyProbs: asFloat32Slice(m["policy_probs"]),
			Value:       asFloat32(m["value"]),
			Body:        zipPoints(bodyX, bodyY),
		}
		snakes = append(snakes, s)
	}
	return snakes
}
