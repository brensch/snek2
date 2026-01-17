package store

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/parquet-go/parquet-go"
	"github.com/parquet-go/parquet-go/compress/zstd"
)

// TrainingRow is a single supervised training sample.
//
// State is a self-contained, raw game state snapshot for a specific (game, turn, ego).
// It is intentionally model-agnostic: trainers can featurize it however they like.
//
// Policy is the action label: 0=Up, 1=Down, 2=Left, 3=Right.
// Value is the outcome target in [-1..1] from the ego snake perspective.
type TrainingRow struct {
	GameID      string  `parquet:"game_id,dict"`
	Turn        int32   `parquet:"turn"`
	EgoID       string  `parquet:"ego_id,dict"`
	Width       int32   `parquet:"width"`
	Height      int32   `parquet:"height"`
	StateFormat string  `parquet:"state_format,dict"`
	State       []byte  `parquet:"state"`
	Policy      int32   `parquet:"policy"`
	Value       float32 `parquet:"value"`
	Source      string  `parquet:"source,dict"`
}

// RawGameState is the canonical, model-agnostic snapshot stored in TrainingRow.State.
// Coordinates follow Battlesnake conventions: (0,0) is bottom-left.
type RawGameState struct {
	Width   int32   `json:"width"`
	Height  int32   `json:"height"`
	Turn    int32   `json:"turn"`
	YouID   string  `json:"you_id"`
	Food    []Point `json:"food"`
	Hazards []Point `json:"hazards,omitempty"`
	Snakes  []Snake `json:"snakes"`
}

type Point struct {
	X int32 `json:"x"`
	Y int32 `json:"y"`
}

type Snake struct {
	ID     string  `json:"id"`
	Health int32   `json:"health"`
	Body   []Point `json:"body"`
}

// ArchiveTurnRow is a single (game, turn) snapshot intended for long-term storage.
//
// It is model-agnostic and optimized for compression:
// - one row per turn (no duplication of food/hazards across snakes)
// - nested/repeated snake data
//
// Policy is the action label for each snake on this turn: 0=Up, 1=Down, 2=Left, 3=Right.
// If unknown or not applicable, Policy is -1.
// Value is the final outcome target in [-1..1] from that snake's perspective.
type ArchiveTurnRow struct {
	GameID string `parquet:"game_id,dict"`
	Turn   int32  `parquet:"turn"`
	Width  int32  `parquet:"width"`
	Height int32  `parquet:"height"`

	FoodX []int32 `parquet:"food_x"`
	FoodY []int32 `parquet:"food_y"`

	HazardX []int32 `parquet:"hazard_x"`
	HazardY []int32 `parquet:"hazard_y"`

	Snakes []ArchiveSnake `parquet:"snakes"`

	Source string `parquet:"source,dict"`

	// ModelPath is the resolved path to the ONNX model used to generate this game.
	// Symlinks are resolved to show the actual model file.
	ModelPath string `parquet:"model_path,dict,optional"`

	// MCTSRootJSON stores a summary of the MCTS root children (joint actions explored).
	// Format: JSON array of {moves: {snakeID: move}, n: visitCount, q: qValue, p: prior}
	// This allows replaying/debugging without the full tree.
	MCTSRootJSON []byte `parquet:"mcts_root_json,optional,zstd"`
}

type ArchiveSnake struct {
	ID     string `parquet:"id,dict"`
	Alive  bool   `parquet:"alive"`
	Health int32  `parquet:"health"`

	BodyX []int32 `parquet:"body_x"`
	BodyY []int32 `parquet:"body_y"`

	Policy int32 `parquet:"policy"`
	// PolicyProbs is an optional distribution target over actions.
	// For self-play games, this is typically the normalized MCTS visit counts.
	PolicyProbs []float32 `parquet:"policy_probs"`
	Value       float32   `parquet:"value"`
}

func WriteArchiveParquet(outPath string, rows []ArchiveTurnRow) error {
	if err := os.MkdirAll(filepath.Dir(outPath), 0o755); err != nil {
		return fmt.Errorf("create output dir: %w", err)
	}

	// Write to a temp file and rename atomically.
	tmpPath := outPath + ".tmp"
	_ = os.Remove(tmpPath)

	if err := parquet.WriteFile(tmpPath, rows,
		parquet.Compression(&zstd.Codec{Level: zstd.SpeedBetterCompression}),
		parquet.KeyValueMetadata("schema", "archive_turn_v1"),
	); err != nil {
		return fmt.Errorf("write parquet: %w", err)
	}

	if err := os.Rename(tmpPath, outPath); err != nil {
		return fmt.Errorf("rename parquet: %w", err)
	}
	return nil
}

func WriteArchiveBatchParquetAtomic(outDir string, rows []ArchiveTurnRow) (string, error) {
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return "", fmt.Errorf("create output dir: %w", err)
	}

	tmpDir := filepath.Join(outDir, "tmp")
	if err := os.MkdirAll(tmpDir, 0o755); err != nil {
		return "", fmt.Errorf("create tmp dir: %w", err)
	}

	name := fmt.Sprintf("batch_%d.parquet", time.Now().UnixNano())
	finalPath := filepath.Join(outDir, name)
	tmpPath := filepath.Join(tmpDir, name+".tmp")
	_ = os.Remove(tmpPath)

	if err := parquet.WriteFile(tmpPath, rows,
		parquet.Compression(&zstd.Codec{Level: zstd.SpeedBetterCompression}),
		parquet.KeyValueMetadata("schema", "archive_turn_v1"),
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

func EncodeRawStateJSON(state RawGameState) ([]byte, error) {
	// Ensure the embedded dimensions are consistent.
	if state.Width <= 0 || state.Height <= 0 {
		return nil, fmt.Errorf("invalid state dimensions: %dx%d", state.Width, state.Height)
	}
	return json.Marshal(state)
}

func WriteGameParquet(outPath string, rows []TrainingRow) error {
	if err := os.MkdirAll(filepath.Dir(outPath), 0o755); err != nil {
		return fmt.Errorf("create output dir: %w", err)
	}

	// Write to a temp file and rename atomically.
	tmpPath := outPath + ".tmp"
	_ = os.Remove(tmpPath)

	// Keep output simple and compact; avoid writing page bounds for raw feature blobs.
	if err := parquet.WriteFile(tmpPath, rows,
		parquet.Compression(&zstd.Codec{Level: zstd.SpeedBetterCompression}),
		parquet.SkipPageBounds("state"),
		parquet.KeyValueMetadata("schema", "raw_training_row_v1"),
	); err != nil {
		return fmt.Errorf("write parquet: %w", err)
	}

	if err := os.Rename(tmpPath, outPath); err != nil {
		return fmt.Errorf("rename parquet: %w", err)
	}
	return nil
}

// WriteBatchParquet writes a batch file containing multiple games.
// The returned path is the final parquet file path.
func WriteBatchParquet(outDir string, rows []TrainingRow) (string, error) {
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return "", fmt.Errorf("create output dir: %w", err)
	}
	name := fmt.Sprintf("batch_%d.parquet", time.Now().UnixNano())
	outPath := filepath.Join(outDir, name)
	if err := WriteGameParquet(outPath, rows); err != nil {
		return "", err
	}
	return outPath, nil
}

// WriteBatchParquetAtomic writes a Parquet file into outDir/tmp and then
// atomically moves it into outDir.
//
// This is useful for long-running writers (like self-play) that want to ensure
// readers never observe partially-written Parquet files.
func WriteBatchParquetAtomic(outDir string, rows []TrainingRow) (string, error) {
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return "", fmt.Errorf("create output dir: %w", err)
	}

	tmpDir := filepath.Join(outDir, "tmp")
	if err := os.MkdirAll(tmpDir, 0o755); err != nil {
		return "", fmt.Errorf("create tmp dir: %w", err)
	}

	name := fmt.Sprintf("batch_%d.parquet", time.Now().UnixNano())
	finalPath := filepath.Join(outDir, name)
	tmpPath := filepath.Join(tmpDir, name+".tmp")
	_ = os.Remove(tmpPath)

	if err := parquet.WriteFile(tmpPath, rows,
		parquet.Compression(&zstd.Codec{Level: zstd.SpeedBetterCompression}),
		parquet.SkipPageBounds("state"),
		parquet.KeyValueMetadata("schema", "raw_training_row_v1"),
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
