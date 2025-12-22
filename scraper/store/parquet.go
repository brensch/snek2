package store

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/parquet-go/parquet-go"
	"github.com/parquet-go/parquet-go/compress/zstd"
)

// TrainingRow is a single supervised training sample.
//
// x is a little-endian float32 tensor flattened as (C,H,W) with C=14, H=W=11.
// policy is the action label: 0=Up, 1=Down, 2=Left, 3=Right.
// value is the outcome target in [-1..1] from the ego snake perspective.
//
// We keep metadata so the trainer can debug and/or filter later.
type TrainingRow struct {
	GameID string  `parquet:"game_id,dict"`
	Turn   int32   `parquet:"turn"`
	EgoID  string  `parquet:"ego_id,dict"`
	Width  int32   `parquet:"width"`
	Height int32   `parquet:"height"`
	X      []byte  `parquet:"x"`
	Policy int32   `parquet:"policy"`
	Value  float32 `parquet:"value"`
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
		parquet.SkipPageBounds("x"),
		parquet.KeyValueMetadata("schema", "training_row_v1"),
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
