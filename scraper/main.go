package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/brensch/snek2/scraper/discovery"
	"github.com/brensch/snek2/scraper/downloader"
	"github.com/brensch/snek2/scraper/store"
)

func main() {
	// Minimal flags (favor simplicity)
	outDir := flag.String("out-dir", getEnvOrDefault("OUT_DIR", "data"), "Directory to write batch .parquet files")
	logPath := flag.String("log-path", getEnvOrDefault("WRITTEN_LOG", "scraper-data/written_games.log"), "Append-only log of game IDs already written")
	flushGames := flag.Int("flush-games", getEnvIntOrDefault("FLUSH_GAMES", 1000), "Flush when buffered games reaches this count")
	flushEvery := flag.Duration("flush-every", getEnvDurationOrDefault("FLUSH_EVERY", 1*time.Hour), "Flush at this interval regardless of buffered count")
	maxPlayers := flag.Int("max-players", getEnvIntOrDefault("MAX_PLAYERS", 50), "Maximum number of players to check from leaderboard")
	requestDelay := flag.Duration("delay", getEnvDurationOrDefault("DELAY", 500*time.Millisecond), "Delay between HTTP requests")

	flag.Parse()

	written, err := store.OpenWrittenLog(*logPath)
	if err != nil {
		log.Fatalf("Failed to open written log: %v", err)
	}
	defer written.Close()

	log.Printf("Starting Battlesnake Scraper (Parquet)")
	log.Printf("  Out Dir: %s", *outDir)
	log.Printf("  Written Log: %s (%d already)", *logPath, written.Count())
	log.Printf("  Flush Games: %d", *flushGames)
	log.Printf("  Flush Every: %s", *flushEvery)
	log.Printf("  Max Players: %d", *maxPlayers)
	log.Printf("  Request Delay: %s", *requestDelay)

	runOnce(written, *outDir, *maxPlayers, *requestDelay, *flushGames, *flushEvery)
}

func runOnce(written *store.WrittenLog, outDir string, maxPlayers int, requestDelay time.Duration, flushEveryGames int, flushEvery time.Duration) {
	if flushEveryGames <= 0 {
		flushEveryGames = 1000
	}
	if flushEvery <= 0 {
		flushEvery = 1 * time.Hour
	}

	if err := os.MkdirAll(outDir, 0o755); err != nil {
		log.Fatalf("Failed to create output dir: %v", err)
	}
	outDir, _ = filepath.Abs(outDir)

	existingIDs := written.SnapshotBoolMap()
	log.Printf("Loaded %d existing game IDs", len(existingIDs))

	// Downloader config (single-threaded)
	dlConfig := downloader.Config{
		EngineURL:      "wss://engine.battlesnake.com/games/%s/events",
		ConnectTimeout: 10 * time.Second,
		ReadTimeout:    30 * time.Second,
	}

	// Start discovery
	discConfig := discovery.Config{
		LeaderboardURLs: []string{
			"https://play.battlesnake.com/leaderboard/standard",
			"https://play.battlesnake.com/leaderboard/standard-duels",
		},
		RequestDelay: requestDelay,
		MaxPlayers:   maxPlayers,
	}
	discWorker := discovery.NewWorker(discConfig, existingIDs)
	gameIDChan := make(chan string, 1000)

	go func() {
		defer close(gameIDChan)
		if err := discWorker.Discover(gameIDChan); err != nil {
			log.Printf("Discovery error: %v", err)
		}
	}()

	flushTicker := time.NewTicker(flushEvery)
	defer flushTicker.Stop()
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	defer signal.Stop(sigChan)

	rowsBuf := make([]store.TrainingRow, 0, 1024)
	gamesBuf := make([]string, 0, flushEveryGames)

	var gamesDownloaded int
	var gamesSkipped int
	var gamesFailed int
	var gamesAttempted int
	var batchesWritten int
	var rowsWritten int

	flush := func(reason string) {
		if len(gamesBuf) == 0 {
			return
		}
		log.Printf("Flushing batch (%s): buffered_games=%d buffered_rows=%d", reason, len(gamesBuf), len(rowsBuf))
		outPath, err := store.WriteBatchParquet(outDir, rowsBuf)
		if err != nil {
			log.Printf("Flush failed (%s): %v", reason, err)
			return
		}
		if err := written.AddMany(gamesBuf); err != nil {
			log.Printf("Flush log append failed (%s): %v", reason, err)
			// We intentionally do NOT treat this as fatal; Parquet is written.
		}
		batchesWritten++
		rowsWritten += len(rowsBuf)
		log.Printf("Flushed batch: games=%d rows=%d path=%s", len(gamesBuf), len(rowsBuf), outPath)

		rowsBuf = rowsBuf[:0]
		gamesBuf = gamesBuf[:0]
	}

	for {
		select {
		case <-sigChan:
			flush("signal")
			log.Printf("Interrupted; exiting")
			return
		case <-flushTicker.C:
			flush("ticker")
		case gameID, ok := <-gameIDChan:
			if !ok {
				flush("final")
				log.Printf("Scraping complete:")
				log.Printf("  Games attempted: %d", gamesAttempted)
				log.Printf("  Games downloaded: %d", gamesDownloaded)
				log.Printf("  Games skipped: %d", gamesSkipped)
				log.Printf("  Games failed: %d", gamesFailed)
				log.Printf("  Batches written: %d", batchesWritten)
				log.Printf("  Rows written: %d", rowsWritten)
				return
			}

			if written.Has(gameID) {
				gamesSkipped++
				continue
			}

			gamesAttempted++
			frames, err := downloader.DownloadGame(gameID, dlConfig)
			if err != nil {
				gamesFailed++
				if gamesFailed%50 == 1 {
					log.Printf("Download failures=%d (latest %s: %v)", gamesFailed, gameID, err)
				}
				continue
			}
			if len(frames) < 2 {
				gamesFailed++
				if gamesFailed%50 == 1 {
					log.Printf("Download failures=%d (latest %s: not enough frames=%d)", gamesFailed, gameID, len(frames))
				}
				continue
			}

			rows := downloader.BuildTrainingRows(gameID, frames)
			if len(rows) == 0 {
				gamesFailed++
				if gamesFailed%50 == 1 {
					log.Printf("Row build failures=%d (latest %s)", gamesFailed, gameID)
				}
				continue
			}

			rowsBuf = append(rowsBuf, rows...)
			gamesBuf = append(gamesBuf, gameID)
			gamesDownloaded++
			if gamesDownloaded%50 == 0 {
				log.Printf("Progress: downloaded=%d skipped=%d failed=%d buffered_games=%d buffered_rows=%d", gamesDownloaded, gamesSkipped, gamesFailed, len(gamesBuf), len(rowsBuf))
			}

			if len(gamesBuf) >= flushEveryGames {
				flush("count")
			}
		}
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
