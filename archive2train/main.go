package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	"github.com/brensch/snek2/executor/convert"
	"github.com/brensch/snek2/game"
	"github.com/brensch/snek2/scraper/store"
	"github.com/parquet-go/parquet-go"
	"github.com/parquet-go/parquet-go/compress/zstd"
)

type TrainingXRow struct {
	GameID string `parquet:"game_id,dict"`
	Turn   int32  `parquet:"turn"`
	EgoID  string `parquet:"ego_id,dict"`

	X []byte `parquet:"x"`

	Policy int32 `parquet:"policy"`
	// Store policy distribution as scalar columns (p0..p3) for better
	// cross-library Parquet compatibility (some readers struggle with LIST<FLOAT>).
	PolicyP0 float32 `parquet:"policy_p0"`
	PolicyP1 float32 `parquet:"policy_p1"`
	PolicyP2 float32 `parquet:"policy_p2"`
	PolicyP3 float32 `parquet:"policy_p3"`
	Value    float32 `parquet:"value"`

	XC int32 `parquet:"x_c"`
	XH int32 `parquet:"x_h"`
	XW int32 `parquet:"x_w"`

	Source string `parquet:"source,dict"`
}

func main() {
	inDir := flag.String("in-dir", "", "Directory containing archive parquet shards")
	outDir := flag.String("out-dir", "", "Output directory for training parquet shards")
	flag.Parse()

	if *inDir == "" || *outDir == "" {
		fmt.Fprintln(os.Stderr, "-in-dir and -out-dir are required")
		os.Exit(2)
	}

	absIn, _ := filepath.Abs(*inDir)
	absOut, _ := filepath.Abs(*outDir)
	if absIn == absOut {
		fmt.Fprintln(os.Stderr, "out-dir must be different from in-dir")
		os.Exit(2)
	}

	if err := os.MkdirAll(absOut, 0o755); err != nil {
		fmt.Fprintf(os.Stderr, "create out-dir: %v\n", err)
		os.Exit(2)
	}

	// Clean old outputs to avoid unbounded growth.
	_ = filepath.WalkDir(absOut, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			return nil
		}
		if strings.HasSuffix(strings.ToLower(d.Name()), ".parquet") {
			_ = os.Remove(path)
		}
		return nil
	})

	inputs := make([]string, 0, 1024)
	_ = filepath.WalkDir(absIn, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
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
	})

	if len(inputs) == 0 {
		fmt.Fprintln(os.Stderr, "no parquet inputs found")
		os.Exit(1)
	}

	convertedFiles := 0
	for _, inPath := range inputs {
		base := filepath.Base(inPath)
		outPath := filepath.Join(absOut, strings.TrimSuffix(base, filepath.Ext(base))+".train.parquet")
		n, err := convertOne(inPath, outPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "convert %s: %v\n", inPath, err)
			continue
		}
		if n > 0 {
			convertedFiles++
		}
	}

	if convertedFiles == 0 {
		fmt.Fprintln(os.Stderr, "no output written (no convertible rows)")
		os.Exit(1)
	}
}

func convertOne(inPath string, outPath string) (int, error) {
	inF, err := os.Open(inPath)
	if err != nil {
		return 0, err
	}
	defer inF.Close()

	reader := parquet.NewGenericReader[store.ArchiveTurnRow](inF)
	defer reader.Close()

	outTmp := outPath + ".tmp"
	_ = os.Remove(outTmp)
	outF, err := os.OpenFile(outTmp, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o644)
	if err != nil {
		return 0, err
	}

	writer := parquet.NewGenericWriter[TrainingXRow](
		outF,
		parquet.Compression(&zstd.Codec{Level: zstd.SpeedBetterCompression}),
	)
	writer.SetKeyValueMetadata("schema", "training_x_row_v2")

	defer func() {
		_ = writer.Close()
		_ = outF.Sync()
		_ = outF.Close()
	}()

	buf := make([]store.ArchiveTurnRow, 256)
	outBuf := make([]TrainingXRow, 0, 2048)
	rowsWritten := 0

	flush := func() error {
		if len(outBuf) == 0 {
			return nil
		}
		if _, err := writer.Write(outBuf); err != nil {
			return err
		}
		rowsWritten += len(outBuf)
		outBuf = outBuf[:0]
		return nil
	}

	for {
		n, err := reader.Read(buf)
		if n > 0 {
			for i := 0; i < n; i++ {
				row := buf[i]
				// Current encoder is fixed-size (11x11). Skip incompatible boards.
				if row.Width != int32(convert.Width) || row.Height != int32(convert.Height) {
					continue
				}

				food := make([]game.Point, 0, min(len(row.FoodX), len(row.FoodY)))
				for j := 0; j < min(len(row.FoodX), len(row.FoodY)); j++ {
					food = append(food, game.Point{X: row.FoodX[j], Y: row.FoodY[j]})
				}

				snakes := make([]game.Snake, 0, len(row.Snakes))
				for _, s := range row.Snakes {
					body := make([]game.Point, 0, min(len(s.BodyX), len(s.BodyY)))
					for k := 0; k < min(len(s.BodyX), len(s.BodyY)); k++ {
						body = append(body, game.Point{X: s.BodyX[k], Y: s.BodyY[k]})
					}
					snakes = append(snakes, game.Snake{Id: s.ID, Health: s.Health, Body: body})
				}

				for _, ego := range row.Snakes {
					if ego.Policy < 0 {
						continue
					}
					st := &game.GameState{
						Width:  row.Width,
						Height: row.Height,
						Turn:   row.Turn,
						YouId:  ego.ID,
						Food:   food,
						Snakes: snakes,
					}

					bPtr := convert.StateToBytes(st)
					x := make([]byte, len(*bPtr))
					copy(x, *bPtr)
					convert.PutBuffer(bPtr)

					policyProbs := ego.PolicyProbs
					if len(policyProbs) != 4 {
						// Some older/selfplay archive shards may omit policy_probs.
						// Fall back to a one-hot distribution derived from the chosen action.
						p := int(ego.Policy)
						if p < 0 || p > 3 {
							return 0, fmt.Errorf("invalid policy (=%d) for game=%s turn=%d ego=%s source=%s file=%s", ego.Policy, row.GameID, row.Turn, ego.ID, row.Source, inPath)
						}
						policyProbs = make([]float32, 4)
						policyProbs[p] = 1.0
					}

					outBuf = append(outBuf, TrainingXRow{
						GameID:   row.GameID,
						Turn:     row.Turn,
						EgoID:    ego.ID,
						X:        x,
						Policy:   ego.Policy,
						PolicyP0: policyProbs[0],
						PolicyP1: policyProbs[1],
						PolicyP2: policyProbs[2],
						PolicyP3: policyProbs[3],
						Value:    ego.Value,
						XC:       int32(convert.Channels),
						XH:       int32(convert.Height),
						XW:       int32(convert.Width),
						Source:   row.Source,
					})

					if len(outBuf) >= 2048 {
						if err := flush(); err != nil {
							return 0, err
						}
					}
				}
			}
		}

		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return 0, err
		}
	}

	if err := flush(); err != nil {
		return 0, err
	}
	if err := writer.Close(); err != nil {
		return 0, err
	}
	if err := outF.Sync(); err != nil {
		return 0, err
	}
	if err := outF.Close(); err != nil {
		return 0, err
	}

	if rowsWritten == 0 {
		_ = os.Remove(outTmp)
		return 0, nil
	}

	if err := os.Rename(outTmp, outPath); err != nil {
		_ = os.Remove(outTmp)
		return 0, err
	}
	return rowsWritten, nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
