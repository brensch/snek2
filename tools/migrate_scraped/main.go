package main

import (
	"flag"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	"github.com/brensch/snek2/scraper/store"
	"github.com/parquet-go/parquet-go"
	"github.com/parquet-go/parquet-go/compress/zstd"
)

func main() {
	inDir := flag.String("in-dir", "processed/scraped", "Directory containing old-schema scraped archive parquet")
	outDir := flag.String("out-dir", "data/scraped", "Output directory for migrated parquet")
	overwrite := flag.Bool("overwrite", false, "Overwrite existing output files")
	flag.Parse()

	absIn, err := filepath.Abs(*inDir)
	if err != nil {
		die("abs in-dir: %v", err)
	}
	absOut, err := filepath.Abs(*outDir)
	if err != nil {
		die("abs out-dir: %v", err)
	}
	if absIn == absOut {
		die("out-dir must be different from in-dir")
	}

	inputs := make([]string, 0, 1024)
	if err := filepath.WalkDir(absIn, func(path string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return nil
		}
		if d.IsDir() {
			name := d.Name()
			if name == "tmp" || name == "processed" || name == "materialized" {
				return filepath.SkipDir
			}
			return nil
		}
		if strings.HasSuffix(strings.ToLower(d.Name()), ".parquet") {
			inputs = append(inputs, path)
		}
		return nil
	}); err != nil {
		die("walk in-dir: %v", err)
	}
	if len(inputs) == 0 {
		die("no parquet inputs found in %s", absIn)
	}

	if err := os.MkdirAll(absOut, 0o755); err != nil {
		die("create out-dir: %v", err)
	}
	tmpDir := filepath.Join(absOut, "tmp")
	if err := os.MkdirAll(tmpDir, 0o755); err != nil {
		die("create tmp dir: %v", err)
	}

	migrated := 0
	skipped := 0
	failed := 0

	for _, inPath := range inputs {
		base := filepath.Base(inPath)
		outPath := filepath.Join(absOut, base)
		tmpOut := filepath.Join(tmpDir, base+".tmp")

		if !*overwrite {
			if _, err := os.Stat(outPath); err == nil {
				skipped++
				continue
			}
		}

		if err := migrateOne(inPath, tmpOut); err != nil {
			failed++
			fmt.Fprintf(os.Stderr, "migrate %s: %v\n", inPath, err)
			_ = os.Remove(tmpOut)
			continue
		}

		_ = os.Remove(outPath)
		if err := os.Rename(tmpOut, outPath); err != nil {
			failed++
			fmt.Fprintf(os.Stderr, "rename %s -> %s: %v\n", tmpOut, outPath, err)
			_ = os.Remove(tmpOut)
			continue
		}

		migrated++
		if migrated%25 == 0 {
			fmt.Fprintf(os.Stderr, "migrated %d/%d...\n", migrated, len(inputs))
		}
	}

	fmt.Fprintf(os.Stderr, "done: migrated=%d skipped=%d failed=%d (in=%d)\n", migrated, skipped, failed, len(inputs))
	if failed > 0 {
		os.Exit(1)
	}
}

func migrateOne(inPath string, outPath string) error {
	inF, err := os.Open(inPath)
	if err != nil {
		return err
	}
	defer inF.Close()

	reader := parquet.NewGenericReader[store.ArchiveTurnRow](inF)
	defer reader.Close()

	outF, err := os.OpenFile(outPath, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer func() {
		_ = outF.Close()
	}()

	writer := parquet.NewGenericWriter[store.ArchiveTurnRow](
		outF,
		parquet.Compression(&zstd.Codec{Level: zstd.SpeedBetterCompression}),
	)
	writer.SetKeyValueMetadata("schema", "archive_turn_v1")
	defer writer.Close()

	buf := make([]store.ArchiveTurnRow, 512)
	for {
		n, readErr := reader.Read(buf)
		if n > 0 {
			for i := 0; i < n; i++ {
				row := &buf[i]
				for j := range row.Snakes {
					s := &row.Snakes[j]
					if s.Policy < 0 || s.Policy > 3 {
						s.Policy = -1
						s.PolicyProbs = nil
						continue
					}
					if len(s.PolicyProbs) == 4 {
						continue
					}
					pp := make([]float32, 4)
					pp[s.Policy] = 1.0
					s.PolicyProbs = pp
				}
			}

			if _, err := writer.Write(buf[:n]); err != nil {
				return err
			}
		}
		if readErr != nil {
			if readErr == io.EOF {
				break
			}
			return readErr
		}
	}

	if err := writer.Close(); err != nil {
		return err
	}
	if err := outF.Sync(); err != nil {
		return err
	}
	return outF.Close()
}

func die(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(2)
}
