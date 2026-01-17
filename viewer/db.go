package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"log"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	_ "github.com/duckdb/duckdb-go/v2"
)

// DBCache maintains a cached DuckDB connection that refreshes periodically.
type DBCache struct {
	roots       []string
	refreshRate time.Duration

	mu          sync.RWMutex
	db          *sql.DB
	lastRefresh time.Time

	// Cached games index for fast pagination
	gamesIndex     []GameSummary
	gamesIndexTime time.Time
}

// NewDBCache creates a new DBCache with the given roots and refresh rate.
func NewDBCache(roots []string, refreshRate time.Duration) *DBCache {
	return &DBCache{
		roots:       roots,
		refreshRate: refreshRate,
	}
}

// Get returns the cached DB connection, refreshing if needed.
func (c *DBCache) Get() (*sql.DB, error) {
	c.mu.RLock()
	if c.db != nil && time.Since(c.lastRefresh) < c.refreshRate {
		db := c.db
		c.mu.RUnlock()
		return db, nil
	}
	c.mu.RUnlock()

	// Need to refresh
	c.mu.Lock()
	defer c.mu.Unlock()

	// Double-check after acquiring write lock
	if c.db != nil && time.Since(c.lastRefresh) < c.refreshRate {
		return c.db, nil
	}

	return c.refreshLocked()
}

// Refresh forces a refresh of the cached DB connection.
func (c *DBCache) Refresh() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	_, err := c.refreshLocked()
	return err
}

func (c *DBCache) refreshLocked() (*sql.DB, error) {
	start := time.Now()

	newDB, err := openDuckDBWithGlobs(c.roots)
	if err != nil {
		return nil, err
	}

	// Close old DB if exists
	if c.db != nil {
		_ = c.db.Close()
	}

	c.db = newDB
	c.lastRefresh = time.Now()
	// Invalidate games index so it gets rebuilt on next access
	c.gamesIndex = nil
	c.gamesIndexTime = time.Time{}

	log.Printf("DBCache refreshed in %v", time.Since(start))
	return c.db, nil
}

// GetGamesIndex returns the cached games index, rebuilding if needed.
// The index is only rebuilt when the DB itself is refreshed (new files detected).
func (c *DBCache) GetGamesIndex(ctx context.Context) ([]GameSummary, error) {
	c.mu.RLock()
	if c.gamesIndex != nil && c.db != nil {
		idx := c.gamesIndex
		c.mu.RUnlock()
		return idx, nil
	}
	c.mu.RUnlock()

	c.mu.Lock()
	defer c.mu.Unlock()

	// Double-check
	if c.gamesIndex != nil && c.db != nil {
		return c.gamesIndex, nil
	}

	// Ensure DB is initialized
	if c.db == nil {
		if _, err := c.refreshLocked(); err != nil {
			return nil, err
		}
	}

	start := time.Now()
	games, err := queryAllGames(ctx, c.db, c.roots)
	if err != nil {
		return nil, err
	}

	c.gamesIndex = games
	c.gamesIndexTime = time.Now()
	log.Printf("Games index rebuilt: %d games in %v", len(games), time.Since(start))

	return c.gamesIndex, nil
}

// Close closes the cached DB connection.
func (c *DBCache) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.db != nil {
		err := c.db.Close()
		c.db = nil
		return err
	}
	return nil
}

// openDuckDBWithGlobs creates a DuckDB connection using glob patterns for the roots.
// This is much faster than enumerating all files.
func openDuckDBWithGlobs(roots []string) (*sql.DB, error) {
	db, err := sql.Open("duckdb", ":memory:")
	if err != nil {
		return nil, err
	}
	// Basic pragmas; ignore errors for compatibility across versions.
	_, _ = db.Exec("PRAGMA threads=4")

	// Build glob patterns for each root, excluding tmp directories
	globs := make([]string, 0, len(roots))
	for _, root := range roots {
		root = strings.TrimSpace(root)
		if root == "" {
			continue
		}
		// Use glob pattern to match all parquet files recursively
		glob := filepath.Join(root, "**", "*.parquet")
		globs = append(globs, "'"+escapeSQLString(glob)+"'")
	}

	if len(globs) == 0 {
		// Empty view
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

	// Use glob patterns directly - DuckDB handles this efficiently
	// Exclude tmp directories by filtering on filename
	// Use union_by_name=true to handle parquet files with different schemas (some may have model_path, some may not)
	sqlText := `CREATE OR REPLACE VIEW turns AS 
		SELECT * FROM read_parquet([` + strings.Join(globs, ",") + `], filename=true, union_by_name=true)
		WHERE NOT contains(filename, '/tmp/')`
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

// queryAllGames loads all game summaries from DuckDB (used to build the cache).
func queryAllGames(ctx context.Context, db *sql.DB, roots []string) ([]GameSummary, error) {
	// Query that gets game summaries AND results in one pass using window functions
	query := `WITH game_stats AS (
		SELECT
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
		GROUP BY game_id
	),
	last_turns AS (
		SELECT game_id, snakes
		FROM (
			SELECT game_id, snakes, turn,
				row_number() OVER (PARTITION BY game_id ORDER BY turn DESC) AS rn
			FROM turns
		)
		WHERE rn = 1
	)
	SELECT 
		g.game_id,
		g.started_ns,
		g.min_turn,
		g.max_turn,
		g.turn_count,
		g.width,
		g.height,
		g.source,
		g.file,
		lt.snakes
	FROM game_stats g
	LEFT JOIN last_turns lt ON g.game_id = lt.game_id`

	rows, err := db.QueryContext(ctx, query)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := make([]GameSummary, 0, 10000)
	for rows.Next() {
		var g GameSummary
		var file string
		var snakesAny any
		if err := rows.Scan(&g.GameID, &g.StartedNs, &g.MinTurn, &g.MaxTurn, &g.TurnCount, &g.Width, &g.Height, &g.Source, &file, &snakesAny); err != nil {
			return nil, err
		}
		g.SourceFile = makeRelativeToRoots(file, roots)
		g.Results = formatSnakeResults(asSnakes(snakesAny))
		out = append(out, g)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}

	// Pre-sort by started_ns DESC (default) for fast subsequent sorts
	sort.Slice(out, func(i, j int) bool {
		// Handle nil started_ns (put at end)
		if out[i].StartedNs == nil && out[j].StartedNs == nil {
			return out[i].GameID > out[j].GameID
		}
		if out[i].StartedNs == nil {
			return false
		}
		if out[j].StartedNs == nil {
			return true
		}
		if *out[i].StartedNs != *out[j].StartedNs {
			return *out[i].StartedNs > *out[j].StartedNs
		}
		return out[i].GameID > out[j].GameID
	})

	return out, nil
}

// paginateGames sorts and paginates a games index in memory.
func paginateGames(games []GameSummary, limit, offset int, sortKey, sortDir string) []GameSummary {
	sk, sd := normalizeSort(sortKey, sortDir)

	// Sort the slice based on the sort key
	sorted := make([]GameSummary, len(games))
	copy(sorted, games)

	sort.Slice(sorted, func(i, j int) bool {
		var less bool
		switch sk {
		case "started_ns":
			// Handle nil
			if sorted[i].StartedNs == nil && sorted[j].StartedNs == nil {
				less = sorted[i].GameID < sorted[j].GameID
			} else if sorted[i].StartedNs == nil {
				less = false // nil goes last in ASC
			} else if sorted[j].StartedNs == nil {
				less = true
			} else {
				less = *sorted[i].StartedNs < *sorted[j].StartedNs
			}
		case "game_id":
			less = sorted[i].GameID < sorted[j].GameID
		case "turn_count":
			less = sorted[i].TurnCount < sorted[j].TurnCount
		case "width":
			less = sorted[i].Width < sorted[j].Width
		case "height":
			less = sorted[i].Height < sorted[j].Height
		case "source":
			less = sorted[i].Source < sorted[j].Source
		case "file":
			less = sorted[i].SourceFile < sorted[j].SourceFile
		default:
			less = sorted[i].GameID < sorted[j].GameID
		}

		if sd == "desc" {
			return !less
		}
		return less
	})

	// Apply pagination
	if offset >= len(sorted) {
		return []GameSummary{}
	}
	end := offset + limit
	if end > len(sorted) {
		end = len(sorted)
	}
	return sorted[offset:end]
}

// queryGamesFromIndex returns paginated games from the cached index.
func queryGamesFromIndex(games []GameSummary, limit, offset int, sortKey, sortDir string) ([]GameSummary, int64) {
	total := int64(len(games))
	paginated := paginateGames(games, limit, offset, sortKey, sortDir)
	return paginated, total
}

// Deprecated: use queryAllGames + paginateGames instead
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
	// Include model_path for selfplay games
	rows, err := db.QueryContext(ctx,
		`SELECT game_id, turn::INTEGER, width::INTEGER, height::INTEGER, food_x, food_y, hazard_x, hazard_y, snakes, source,
		        COALESCE(model_path, '') as model_path
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
		var modelPath string
		if err := rows.Scan(&t.GameID, &t.Turn, &t.Width, &t.Height, &foodXAny, &foodYAny, &hazXAny, &hazYAny, &snakesAny, &t.Source, &modelPath); err != nil {
			return nil, err
		}
		t.Food = zipPoints(asInt32Slice(foodXAny), asInt32Slice(foodYAny))
		t.Hazards = zipPoints(asInt32Slice(hazXAny), asInt32Slice(hazYAny))
		t.Snakes = asSnakes(snakesAny)
		t.ModelPath = modelPath
		turns = append(turns, t)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return turns, nil
}

func queryTurn(ctx context.Context, db *sql.DB, gameID string, turn int32) (Turn, error) {
	// Try to get mcts_root_json and model_path if they exist (selfplay games have them)
	row := db.QueryRowContext(ctx,
		`SELECT game_id, turn::INTEGER, width::INTEGER, height::INTEGER, food_x, food_y, hazard_x, hazard_y, snakes, source,
		        COALESCE(mcts_root_json, NULL) as mcts_root_json,
		        COALESCE(model_path, '') as model_path
		 FROM turns
		 WHERE game_id = ? AND turn = ?
		 LIMIT 1`, gameID, turn)

	var t Turn
	var foodXAny any
	var foodYAny any
	var hazXAny any
	var hazYAny any
	var snakesAny any
	var mctsRootJSON []byte
	var modelPath string
	if err := row.Scan(&t.GameID, &t.Turn, &t.Width, &t.Height, &foodXAny, &foodYAny, &hazXAny, &hazYAny, &snakesAny, &t.Source, &mctsRootJSON, &modelPath); err != nil {
		// Fallback query without mcts_root_json and model_path for older parquet files
		row = db.QueryRowContext(ctx,
			`SELECT game_id, turn::INTEGER, width::INTEGER, height::INTEGER, food_x, food_y, hazard_x, hazard_y, snakes, source
			 FROM turns
			 WHERE game_id = ? AND turn = ?
			 LIMIT 1`, gameID, turn)
		if err := row.Scan(&t.GameID, &t.Turn, &t.Width, &t.Height, &foodXAny, &foodYAny, &hazXAny, &hazYAny, &snakesAny, &t.Source); err != nil {
			return Turn{}, err
		}
	}
	t.Food = zipPoints(asInt32Slice(foodXAny), asInt32Slice(foodYAny))
	t.Hazards = zipPoints(asInt32Slice(hazXAny), asInt32Slice(hazYAny))
	t.Snakes = asSnakes(snakesAny)
	t.ModelPath = modelPath

	// Parse MCTS root JSON if present
	if len(mctsRootJSON) > 0 {
		var rootChildren []JointChildSummary
		if err := json.Unmarshal(mctsRootJSON, &rootChildren); err == nil {
			t.MCTSRoot = rootChildren
		}
	}

	return t, nil
}
