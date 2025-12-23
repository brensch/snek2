package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/brensch/snek2/scraper/discovery"
	"github.com/brensch/snek2/scraper/downloader"
	"github.com/brensch/snek2/scraper/logging"
	"github.com/brensch/snek2/scraper/store"
)

func main() {
	// Minimal flags (favor simplicity)
	outDir := flag.String("out-dir", getEnvOrDefault("OUT_DIR", "data/scraped"), "Directory to write batch .parquet files")
	logPath := flag.String("log-path", getEnvOrDefault("WRITTEN_LOG", "scraper-data/written_games.log"), "Append-only log of game IDs already written")
	logLevel := flag.String("log-level", getEnvOrDefault("LOG_LEVEL", "info"), "Log level: debug|info|warn|error")
	flushGames := flag.Int("flush-games", getEnvIntOrDefault("FLUSH_GAMES", 50), "Flush when buffered games reaches this count")
	flushEvery := flag.Duration("flush-every", getEnvDurationOrDefault("FLUSH_EVERY", 1*time.Hour), "Flush at this interval regardless of buffered count")
	pollEvery := flag.Duration("poll-every", getEnvDurationOrDefault("POLL_EVERY", 1*time.Hour), "How often to re-scan leaderboards for new game IDs")
	requestDelay := flag.Duration("delay", getEnvDurationOrDefault("DELAY", 50*time.Millisecond), "Delay between HTTP requests")

	flag.Parse()

	logger := slog.New(logging.NewPrettyJSONHandler(os.Stdout, &slog.HandlerOptions{AddSource: true, Level: parseLogLevel(*logLevel)}))
	slog.SetDefault(logger)

	written, err := store.OpenWrittenLog(*logPath)
	if err != nil {
		slog.Error("failed to open written log", "error", err, "log_path", *logPath)
		os.Exit(1)
	}
	defer written.Close()

	slog.Info("starting scraper",
		"out_dir", *outDir,
		"written_log", *logPath,
		"written_count", written.Count(),
		"log_level", *logLevel,
		"flush_games", *flushGames,
		"flush_every", flushEvery.String(),
		"poll_every", pollEvery.String(),
		"request_delay", requestDelay.String(),
	)

	runLoop(written, *outDir, *requestDelay, *flushGames, *flushEvery, *pollEvery)
}

func runLoop(written *store.WrittenLog, outDir string, requestDelay time.Duration, flushEveryGames int, flushEvery time.Duration, pollEvery time.Duration) {
	if flushEveryGames <= 0 {
		flushEveryGames = 1000
	}
	if flushEvery <= 0 {
		flushEvery = 1 * time.Hour
	}
	if pollEvery <= 0 {
		pollEvery = 1 * time.Hour
	}

	if err := os.MkdirAll(outDir, 0o755); err != nil {
		slog.Error("failed to create output dir", "error", err, "out_dir", outDir)
		os.Exit(1)
	}
	outDir, _ = filepath.Abs(outDir)

	existingIDs := written.SnapshotBoolMap()
	slog.Info("loaded written ids", "count", len(existingIDs))

	// Downloader config (single-threaded)
	dlConfig := downloader.Config{
		EngineURL:      "wss://engine.battlesnake.com/games/%s/events",
		ConnectTimeout: 10 * time.Second,
		ReadTimeout:    30 * time.Second,
	}

	// Start discovery loop
	discConfig := discovery.Config{
		LeaderboardURLs: []string{
			"https://play.battlesnake.com/leaderboard/standard",
			"https://play.battlesnake.com/leaderboard/standard-duels",
		},
		RequestDelay: requestDelay,
		MaxPlayers:   0,
	}
	discWorker := discovery.NewWorker(discConfig, existingIDs)
	gameIDChan := make(chan string, 10_000)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	go func() {
		runDiscovery := func(reason string) {
			start := time.Now()
			slog.Info("discovery poll starting", "reason", reason)
			stats, err := discWorker.Discover(ctx, gameIDChan)
			if err != nil {
				if ctx.Err() != nil {
					return
				}
				slog.Error("discovery poll error", "reason", reason, "error", err)
				return
			}
			slog.Info("discovery poll finished",
				"reason", reason,
				"duration_ms", time.Since(start).Milliseconds(),
				"leaderboards", stats.Leaderboards,
				"players", stats.Players,
				"game_links", stats.GameLinks,
				"new_game_ids", stats.NewGameIDs,
				"known_game_ids", stats.KnownGameIDs,
			)
		}
		runDiscovery("startup")

		ticker := time.NewTicker(pollEvery)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				runDiscovery("ticker")
			}
		}
	}()

	flushTicker := time.NewTicker(flushEvery)
	defer flushTicker.Stop()

	batchWriter, err := store.NewBatchWriter(outDir)
	if err != nil {
		slog.Error("failed to create batch writer", "error", err)
		os.Exit(1)
	}
	slog.Info("opened batch file", "tmp_path", batchWriter.TmpPath())
	batchGameIDs := make([]string, 0, flushEveryGames)

	var gamesDownloaded int
	var gamesSkipped int
	var gamesFailed int
	var gamesAttempted int
	var gameIDsReceived int
	var batchesWritten int
	var rowsWritten int

	flush := func(reason string) {
		if batchWriter.BufferedRows() == 0 {
			return
		}

		slog.Info("flush starting",
			"reason", reason,
			"tmp_path", batchWriter.TmpPath(),
			"buffered_games", batchWriter.BufferedGames(),
			"buffered_rows", batchWriter.BufferedRows(),
		)

		outPath, rows, games, err := batchWriter.Finalize()
		if err != nil {
			slog.Error("flush finalize failed", "reason", reason, "error", err)
			os.Exit(1)
		}
		if outPath == "" {
			// Nothing written; open a new tmp file.
			if reason != "signal" && reason != "final" {
				bw, err := store.NewBatchWriter(outDir)
				if err != nil {
					slog.Error("failed to create batch writer", "error", err)
					os.Exit(1)
				}
				batchWriter = bw
				slog.Info("opened batch file", "tmp_path", batchWriter.TmpPath())
				batchGameIDs = batchGameIDs[:0]
			}
			return
		}

		if err := written.AddMany(batchGameIDs); err != nil {
			slog.Error("flush written-log append failed", "reason", reason, "error", err)
			// Intentionally not fatal: parquet file is already finalized.
		}

		batchesWritten++
		rowsWritten += rows
		slog.Info("flush finished",
			"reason", reason,
			"out_path", outPath,
			"games", games,
			"rows", rows,
			"batches_written", batchesWritten,
			"rows_written", rowsWritten,
		)

		if reason != "signal" && reason != "final" {
			bw, err := store.NewBatchWriter(outDir)
			if err != nil {
				slog.Error("failed to create batch writer", "error", err)
				os.Exit(1)
			}
			batchWriter = bw
			batchGameIDs = batchGameIDs[:0]
			slog.Info("opened batch file", "tmp_path", batchWriter.TmpPath())
		}
	}

	// Main loop.
mainLoop:
	for {
		select {
		case <-ctx.Done():
			break mainLoop
		case <-flushTicker.C:
			flush("ticker")
		case gameID, ok := <-gameIDChan:
			if !ok {
				// Should never happen (discovery loop does not close this channel).
				slog.Error("game id channel closed unexpectedly; exiting")
				flush("final")
				return
			}
			gameIDsReceived++

			if written.Has(gameID) {
				gamesSkipped++
				slog.Debug("game skipped",
					"game_id", gameID,
					"reason", "already_written",
					"skipped", gamesSkipped,
					"downloaded", gamesDownloaded,
					"failed", gamesFailed,
				)
				continue
			}

			gamesAttempted++
			frames, err := downloader.DownloadGame(ctx, gameID, dlConfig)
			if err != nil {
				if ctx.Err() != nil {
					break mainLoop
				}
				gamesFailed++
				if gamesFailed%50 == 1 {
					slog.Warn("download failures",
						"failed", gamesFailed,
						"latest_game_id", gameID,
						"error", err,
					)
				}
				continue
			}
			if len(frames) < 2 {
				gamesFailed++
				if gamesFailed%50 == 1 {
					slog.Warn("download produced too few frames",
						"failed", gamesFailed,
						"latest_game_id", gameID,
						"frames", len(frames),
					)
				}
				continue
			}

			rows := downloader.BuildTrainingRows(gameID, frames)
			if len(rows) == 0 {
				gamesFailed++
				if gamesFailed%50 == 1 {
					slog.Warn("row build failures",
						"failed", gamesFailed,
						"latest_game_id", gameID,
					)
				}
				continue
			}

			if err := batchWriter.WriteRows(rows); err != nil {
				slog.Error("failed to write parquet rows", "error", err, "game_id", gameID, "tmp_path", batchWriter.TmpPath())
				os.Exit(1)
			}
			batchWriter.NoteGameWritten()
			batchGameIDs = append(batchGameIDs, gameID)
			gamesDownloaded++

			slog.Info("game downloaded",
				"game_id", gameID,
				"frames", len(frames),
				"rows", len(rows),
				"tmp_path", batchWriter.TmpPath(),
				"batch_games", batchWriter.BufferedGames(),
				"batch_rows", batchWriter.BufferedRows(),
				"downloaded", gamesDownloaded,
			)

			if batchWriter.BufferedGames() >= flushEveryGames {
				flush("count")
			}
		}
	}

	// Graceful shutdown path.
	flush("signal")
	slog.Info("shutdown complete")
}

func parseLogLevel(s string) slog.Leveler {
	switch s {
	case "debug", "DEBUG":
		return slog.LevelDebug
	case "info", "INFO", "":
		return slog.LevelInfo
	case "warn", "warning", "WARN", "WARNING":
		return slog.LevelWarn
	case "error", "ERROR":
		return slog.LevelError
	default:
		// Fall back to info on unknown values.
		return slog.LevelInfo
	}
}

// Environment variable helpers
func getEnvOrDefault(key, defaultVal string) string {
	if val := os.Getenv(key); val != "" {
		return val
	}
	return defaultVal
}

func getEnvIntOrDefault(key string, defaultVal int) int {
	if val := os.Getenv(key); val != "" {
		var i int
		if _, err := fmt.Sscanf(val, "%d", &i); err == nil {
			return i
		}
	}
	return defaultVal
}

func getEnvDurationOrDefault(key string, defaultVal time.Duration) time.Duration {
	if val := os.Getenv(key); val != "" {
		if d, err := time.ParseDuration(val); err == nil {
			return d
		}
	}
	return defaultVal
}

func getEnvBoolOrDefault(key string, defaultVal bool) bool {
	if val := os.Getenv(key); val != "" {
		return val == "true" || val == "1" || val == "yes"
	}
	return defaultVal
}
