package main

import (
	"context"
	"database/sql"
	"io/fs"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	_ "github.com/duckdb/duckdb-go/v2"
)

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
