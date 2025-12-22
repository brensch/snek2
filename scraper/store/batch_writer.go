package store

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/parquet-go/parquet-go"
	"github.com/parquet-go/parquet-go/compress/zstd"
)

type BatchWriter struct {
	outDir string
	tmpDir string

	name    string
	tmpPath string
	outPath string

	file   *os.File
	writer *parquet.GenericWriter[TrainingRow]

	bufferedGames int
	bufferedRows  int
}

func NewBatchWriter(outDir string) (*BatchWriter, error) {
	if outDir == "" {
		return nil, fmt.Errorf("outDir is required")
	}

	absOut, err := filepath.Abs(outDir)
	if err != nil {
		absOut = outDir
	}
	tmpDir := filepath.Join(absOut, "tmp")
	if err := os.MkdirAll(tmpDir, 0o755); err != nil {
		return nil, fmt.Errorf("create tmp dir: %w", err)
	}

	name := fmt.Sprintf("batch_%d.parquet", time.Now().UnixNano())
	tmpPath := filepath.Join(tmpDir, name)
	outPath := filepath.Join(absOut, name)

	f, err := os.OpenFile(tmpPath, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o644)
	if err != nil {
		return nil, fmt.Errorf("open tmp parquet: %w", err)
	}

	w := parquet.NewGenericWriter[TrainingRow](
		f,
		parquet.Compression(&zstd.Codec{Level: zstd.SpeedBetterCompression}),
		parquet.SkipPageBounds("x"),
	)
	w.SetKeyValueMetadata("schema", "training_row_v1")

	return &BatchWriter{
		outDir:  absOut,
		tmpDir:  tmpDir,
		name:    name,
		tmpPath: tmpPath,
		outPath: outPath,
		file:    f,
		writer:  w,
	}, nil
}

func (b *BatchWriter) TmpPath() string    { return b.tmpPath }
func (b *BatchWriter) OutPath() string    { return b.outPath }
func (b *BatchWriter) BufferedGames() int { return b.bufferedGames }
func (b *BatchWriter) BufferedRows() int  { return b.bufferedRows }

func (b *BatchWriter) WriteRows(rows []TrainingRow) error {
	if b.writer == nil || b.file == nil {
		return fmt.Errorf("batch writer is closed")
	}
	if len(rows) == 0 {
		return nil
	}
	_, err := b.writer.Write(rows)
	if err != nil {
		return err
	}
	b.bufferedRows += len(rows)
	return nil
}

func (b *BatchWriter) NoteGameWritten() {
	b.bufferedGames++
}

// Finalize closes the parquet writer and moves the file from tmp/ to outDir.
// If no rows were written, the tmp file is removed and outPath is returned empty.
func (b *BatchWriter) Finalize() (outPath string, rows int, games int, err error) {
	if b.writer == nil && b.file == nil {
		return "", 0, 0, nil
	}

	rows = b.bufferedRows
	games = b.bufferedGames
	outPath = b.outPath

	var closeErr error
	if b.writer != nil {
		closeErr = b.writer.Close()
		b.writer = nil
	}
	var fileErr error
	if b.file != nil {
		_ = b.file.Sync()
		fileErr = b.file.Close()
		b.file = nil
	}
	if closeErr != nil {
		return "", 0, 0, fmt.Errorf("close parquet writer: %w", closeErr)
	}
	if fileErr != nil {
		return "", 0, 0, fmt.Errorf("close parquet file: %w", fileErr)
	}

	if rows == 0 {
		_ = os.Remove(b.tmpPath)
		return "", 0, 0, nil
	}
	if err := os.Rename(b.tmpPath, b.outPath); err != nil {
		return "", 0, 0, fmt.Errorf("rename parquet: %w", err)
	}
	return outPath, rows, games, nil
}
