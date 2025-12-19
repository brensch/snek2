package db

import (
	"database/sql"
	"fmt"
	"log"
	"sync"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

// DB wraps the SQLite connection with thread-safe operations
type DB struct {
	conn *sql.DB
	mu   sync.Mutex
}

// Game represents a scraped game record
type Game struct {
	ID          string
	Winner      string
	Ruleset     string
	CrawledAt   time.Time
	IsProcessed bool
}

// Frame represents a single turn of game data
type Frame struct {
	GameID  string
	Turn    int
	RawJSON string
}

// New creates a new database connection and initializes the schema
func New(dbPath string) (*DB, error) {
	conn, err := sql.Open("sqlite3", dbPath+"?_journal_mode=WAL&_synchronous=NORMAL")
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	// Enable connection pooling settings
	conn.SetMaxOpenConns(1) // SQLite only supports one writer
	conn.SetMaxIdleConns(1)

	db := &DB{conn: conn}
	if err := db.initSchema(); err != nil {
		conn.Close()
		return nil, err
	}

	return db, nil
}

// initSchema creates the required tables if they don't exist
func (db *DB) initSchema() error {
	schema := `
	-- Table to track unique games and avoid re-downloading
	CREATE TABLE IF NOT EXISTS games (
		id TEXT PRIMARY KEY,           -- The UUID (e.g., "da3de01a...")
		winner TEXT,                   -- Name of the winning snake
		ruleset TEXT,                  -- e.g., "standard"
		crawled_at DATETIME DEFAULT CURRENT_TIMESTAMP,
		is_processed BOOLEAN DEFAULT 0 -- Flag for training converter
	);

	-- Table to store the raw data for every turn
	CREATE TABLE IF NOT EXISTS frames (
		game_id TEXT,
		turn INTEGER,
		raw_json TEXT,                 -- The full JSON blob from the websocket
		PRIMARY KEY (game_id, turn),
		FOREIGN KEY(game_id) REFERENCES games(id)
	);

	-- Index for faster lookups
	CREATE INDEX IF NOT EXISTS idx_games_is_processed ON games(is_processed);
	CREATE INDEX IF NOT EXISTS idx_frames_game_id ON frames(game_id);
	`

	db.mu.Lock()
	defer db.mu.Unlock()

	_, err := db.conn.Exec(schema)
	if err != nil {
		return fmt.Errorf("failed to create schema: %w", err)
	}

	return nil
}

// GameExists checks if a game has already been downloaded
func (db *DB) GameExists(gameID string) (bool, error) {
	db.mu.Lock()
	defer db.mu.Unlock()

	var exists int
	err := db.conn.QueryRow("SELECT 1 FROM games WHERE id = ?", gameID).Scan(&exists)
	if err == sql.ErrNoRows {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	return true, nil
}

// InsertGame inserts a game and all its frames in a single transaction
func (db *DB) InsertGame(game Game, frames []Frame) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	tx, err := db.conn.Begin()
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer tx.Rollback()

	// Insert game
	_, err = tx.Exec(
		"INSERT OR IGNORE INTO games (id, winner, ruleset) VALUES (?, ?, ?)",
		game.ID, game.Winner, game.Ruleset,
	)
	if err != nil {
		return fmt.Errorf("failed to insert game: %w", err)
	}

	// Prepare frame insert statement
	stmt, err := tx.Prepare("INSERT OR IGNORE INTO frames (game_id, turn, raw_json) VALUES (?, ?, ?)")
	if err != nil {
		return fmt.Errorf("failed to prepare frame statement: %w", err)
	}
	defer stmt.Close()

	// Insert all frames
	for _, frame := range frames {
		_, err = stmt.Exec(frame.GameID, frame.Turn, frame.RawJSON)
		if err != nil {
			return fmt.Errorf("failed to insert frame %d: %w", frame.Turn, err)
		}
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit transaction: %w", err)
	}

	return nil
}

// GetUnprocessedGames returns games that haven't been converted to training data
func (db *DB) GetUnprocessedGames(limit int) ([]Game, error) {
	db.mu.Lock()
	defer db.mu.Unlock()

	rows, err := db.conn.Query(
		"SELECT id, winner, ruleset, crawled_at, is_processed FROM games WHERE is_processed = 0 LIMIT ?",
		limit,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var games []Game
	for rows.Next() {
		var g Game
		if err := rows.Scan(&g.ID, &g.Winner, &g.Ruleset, &g.CrawledAt, &g.IsProcessed); err != nil {
			return nil, err
		}
		games = append(games, g)
	}

	return games, rows.Err()
}

// GetGameFrames returns all frames for a specific game
func (db *DB) GetGameFrames(gameID string) ([]Frame, error) {
	db.mu.Lock()
	defer db.mu.Unlock()

	rows, err := db.conn.Query(
		"SELECT game_id, turn, raw_json FROM frames WHERE game_id = ? ORDER BY turn",
		gameID,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var frames []Frame
	for rows.Next() {
		var f Frame
		if err := rows.Scan(&f.GameID, &f.Turn, &f.RawJSON); err != nil {
			return nil, err
		}
		frames = append(frames, f)
	}

	return frames, rows.Err()
}

// MarkGameProcessed marks a game as processed for training
func (db *DB) MarkGameProcessed(gameID string) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	_, err := db.conn.Exec("UPDATE games SET is_processed = 1 WHERE id = ?", gameID)
	return err
}

// Stats returns statistics about the database
func (db *DB) Stats() (totalGames, processedGames, totalFrames int64, err error) {
	db.mu.Lock()
	defer db.mu.Unlock()

	err = db.conn.QueryRow("SELECT COUNT(*) FROM games").Scan(&totalGames)
	if err != nil {
		return
	}

	err = db.conn.QueryRow("SELECT COUNT(*) FROM games WHERE is_processed = 1").Scan(&processedGames)
	if err != nil {
		return
	}

	err = db.conn.QueryRow("SELECT COUNT(*) FROM frames").Scan(&totalFrames)
	return
}

// Close closes the database connection
func (db *DB) Close() error {
	return db.conn.Close()
}

// GetAllGameIDs returns all game IDs in the database
func (db *DB) GetAllGameIDs() (map[string]bool, error) {
	db.mu.Lock()
	defer db.mu.Unlock()

	rows, err := db.conn.Query("SELECT id FROM games")
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	ids := make(map[string]bool)
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			log.Printf("Error scanning game ID: %v", err)
			continue
		}
		ids[id] = true
	}

	return ids, rows.Err()
}
