package store

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/parquet-go/parquet-go"
	"github.com/parquet-go/parquet-go/compress/zstd"
)

// DebugMCTSNode represents a node in the MCTS tree for debug visualization.
// It stores the full tree structure including all explored paths.
type DebugMCTSNode struct {
	// VisitCount is N for this node
	VisitCount int32 `parquet:"visit_count" json:"visit_count"`
	// ValueSum is the sum of values for this node
	ValueSum float32 `parquet:"value_sum" json:"value_sum"`
	// PriorProb is P(a|s) from the policy network
	PriorProb float32 `parquet:"prior_prob" json:"prior_prob"`

	// Move is the action that led to this node (0=Up, 1=Down, 2=Left, 3=Right)
	Move int32 `parquet:"move" json:"move"`
	// UCB is the Upper Confidence Bound at selection time
	UCB float32 `parquet:"ucb" json:"ucb"`

	// State snapshot for this node
	Width       int32    `parquet:"width" json:"width"`
	Height      int32    `parquet:"height" json:"height"`
	Turn        int32    `parquet:"turn" json:"turn"`
	YouId       string   `parquet:"you_id,dict" json:"you_id"`
	FoodX       []int32  `parquet:"food_x" json:"food_x"`
	FoodY       []int32  `parquet:"food_y" json:"food_y"`
	SnakeID     []string `parquet:"snake_id" json:"snake_id"`
	SnakeHealth []int32  `parquet:"snake_health" json:"snake_health"`
	// Flattened body coords per snake: SnakeBodyX[i] corresponds to snake i
	SnakeBodyX [][]int32 `parquet:"snake_body_x" json:"snake_body_x"`
	SnakeBodyY [][]int32 `parquet:"snake_body_y" json:"snake_body_y"`

	// Children stores child nodes (up to 4 moves, nested recursively)
	// Serialized as JSON within parquet for nested tree structure
	ChildrenJSON []byte `parquet:"children_json" json:"-"`
}

// DebugSnakeTree represents the MCTS tree for one snake at a given turn.
type DebugSnakeTree struct {
	SnakeID  string `parquet:"snake_id,dict" json:"snake_id"`
	SnakeIdx int32  `parquet:"snake_idx" json:"snake_idx"`
	// BestMove is the move selected by MCTS (argmax N)
	BestMove int32 `parquet:"best_move" json:"best_move"`
	// RootJSON is the root node serialized as JSON (contains full tree)
	RootJSON []byte `parquet:"root_json" json:"root_json"`
}

// DebugTurnRow stores the full MCTS trees for all snakes at a single turn.
type DebugTurnRow struct {
	GameID    string `parquet:"game_id,dict" json:"game_id"`
	ModelPath string `parquet:"model_path,dict" json:"model_path"`
	Turn      int32  `parquet:"turn" json:"turn"`
	Width     int32  `parquet:"width" json:"width"`
	Height    int32  `parquet:"height" json:"height"`

	// Game state at this turn
	FoodX   []int32 `parquet:"food_x" json:"food_x"`
	FoodY   []int32 `parquet:"food_y" json:"food_y"`
	HazardX []int32 `parquet:"hazard_x" json:"hazard_x"`
	HazardY []int32 `parquet:"hazard_y" json:"hazard_y"`

	// Snakes at this turn
	Snakes []DebugSnake `parquet:"snakes" json:"snakes"`

	// MCTS trees for each snake (serialized as JSON for nested structure)
	TreesJSON []byte `parquet:"trees_json" json:"trees_json"`

	// Simulations and cpuct used
	Sims  int32   `parquet:"sims" json:"sims"`
	Cpuct float32 `parquet:"cpuct" json:"cpuct"`
}

// DebugSnake is similar to ArchiveSnake but for debug games.
type DebugSnake struct {
	ID     string  `parquet:"id,dict" json:"id"`
	Alive  bool    `parquet:"alive" json:"alive"`
	Health int32   `parquet:"health" json:"health"`
	BodyX  []int32 `parquet:"body_x" json:"body_x"`
	BodyY  []int32 `parquet:"body_y" json:"body_y"`
	// The move chosen by MCTS for this snake
	Policy int32 `parquet:"policy" json:"policy"`
}

// DebugGameMeta contains metadata about a debug game.
type DebugGameMeta struct {
	GameID    string  `parquet:"game_id,dict" json:"game_id"`
	ModelPath string  `parquet:"model_path,dict" json:"model_path"`
	CreatedNs int64   `parquet:"created_ns" json:"created_ns"`
	TurnCount int32   `parquet:"turn_count" json:"turn_count"`
	Width     int32   `parquet:"width" json:"width"`
	Height    int32   `parquet:"height" json:"height"`
	Sims      int32   `parquet:"sims" json:"sims"`
	Cpuct     float32 `parquet:"cpuct" json:"cpuct"`
	Winner    string  `parquet:"winner,dict" json:"winner"`
}

// WriteDebugGameParquet writes a debug game to a parquet file.
func WriteDebugGameParquet(outDir string, gameID string, rows []DebugTurnRow) (string, error) {
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return "", fmt.Errorf("create output dir: %w", err)
	}

	name := fmt.Sprintf("debug_%s_%d.parquet", gameID, time.Now().UnixNano())
	finalPath := filepath.Join(outDir, name)
	tmpPath := finalPath + ".tmp"
	_ = os.Remove(tmpPath)

	if err := parquet.WriteFile(tmpPath, rows,
		parquet.Compression(&zstd.Codec{Level: zstd.SpeedBetterCompression}),
		parquet.KeyValueMetadata("schema", "debug_game_v1"),
	); err != nil {
		_ = os.Remove(tmpPath)
		return "", fmt.Errorf("write parquet: %w", err)
	}

	if err := os.Rename(tmpPath, finalPath); err != nil {
		_ = os.Remove(tmpPath)
		return "", fmt.Errorf("rename parquet: %w", err)
	}

	return finalPath, nil
}
