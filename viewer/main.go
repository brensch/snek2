package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"flag"
	"io/fs"
	"log"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

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
}

type GamesResponse struct {
	Total int64         `json:"total"`
	Games []GameSummary `json:"games"`
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

func main() {
	fs := flag.NewFlagSet(os.Args[0], flag.ContinueOnError)
	fs.SetOutput(os.Stderr)

	listen := fs.String("listen", "127.0.0.1:8080", "HTTP listen address")
	dataDir := fs.String("data-dir", "", "Directory containing archive parquet shards (archive_turn_v1) [deprecated: prefer -data-dirs]")
	dataDirs := fs.String("data-dirs", strings.Join(defaultDataDirs(), ","), "Comma-separated list of directories containing archive parquet shards (archive_turn_v1)")
	staticDir := fs.String("static-dir", "", "Optional directory to serve as SPA static (e.g. viewer/web/dist)")
	if err := fs.Parse(os.Args[1:]); err != nil {
		log.Fatalf("flag parse: %v", err)
	}

	roots := parseDataRoots(*dataDirs)
	if strings.TrimSpace(*dataDir) != "" {
		// Back-compat: if user explicitly sets -data-dir, use only that.
		roots = []string{strings.TrimSpace(*dataDir)}
	}

	parquetFiles, err := findParquetFilesMulti(roots)
	if err != nil {
		log.Fatalf("scan parquet files: %v", err)
	}
	log.Printf("Found %d parquet files under %s", len(parquetFiles), strings.Join(roots, ","))

	db, err := openDuckDB(parquetFiles)
	if err != nil {
		log.Fatalf("open duckdb: %v", err)
	}
	defer db.Close()

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

func openDuckDB(parquetFiles []string) (*sql.DB, error) {
	db, err := sql.Open("duckdb", ":memory:")
	if err != nil {
		return nil, err
	}
	// Basic pragmas; ignore errors for compatibility across versions.
	_, _ = db.Exec("PRAGMA threads=4")
	_, _ = db.Exec("PRAGMA enable_object_cache=true")

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
	return out, rows.Err()
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
			for i := range l2 {
				list[i] = l2[i]
			}
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
