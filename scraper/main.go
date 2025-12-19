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

	pb "github.com/brensch/snek2/gen/go"
	"github.com/brensch/snek2/scraper/db"
	"github.com/brensch/snek2/scraper/discovery"
	"github.com/brensch/snek2/scraper/downloader"
	"github.com/brensch/snek2/scraper/exporter"
	"google.golang.org/protobuf/proto"
)

func main() {
	// Command line flags
	dbPath := flag.String("db", getEnvOrDefault("DB_PATH", "battlesnake.db"), "Path to SQLite database")
	numWorkers := flag.Int("workers", getEnvIntOrDefault("WORKERS", 4), "Number of download workers")
	maxPlayers := flag.Int("max-players", getEnvIntOrDefault("MAX_PLAYERS", 50), "Maximum number of players to check from leaderboard")
	requestDelay := flag.Duration("delay", getEnvDurationOrDefault("DELAY", 500*time.Millisecond), "Delay between HTTP requests")
	daemon := flag.Bool("daemon", false, "Run as daemon (continuously check for new games)")
	daemonInterval := flag.Duration("interval", getEnvDurationOrDefault("INTERVAL", 30*time.Minute), "Interval between discovery runs in daemon mode")
	exportMode := flag.Bool("export", false, "Export unprocessed games to training data")
	exportPath := flag.String("export-path", "data/scraped_training.pb", "Output path for exported training data")
	exportMax := flag.Int("export-max", 100, "Maximum games to export")
	statsOnly := flag.Bool("stats", false, "Only show database statistics")
	autoExport := flag.Bool("auto-export", getEnvBoolOrDefault("AUTO_EXPORT", false), "Auto-export after each scrape run (daemon mode)")
	outputDir := flag.String("output-dir", getEnvOrDefault("OUTPUT_DIR", "output"), "Output directory for auto-exported .pb files")

	flag.Parse()

	// Initialize database
	database, err := db.New(*dbPath)
	if err != nil {
		log.Fatalf("Failed to initialize database: %v", err)
	}
	defer database.Close()

	// Stats mode
	if *statsOnly {
		printStats(database)
		return
	}

	// Export mode
	if *exportMode {
		if err := exportTrainingData(database, *exportMax, *exportPath); err != nil {
			log.Fatalf("Export failed: %v", err)
		}
		return
	}

	// Scraping mode
	log.Printf("Starting Battlesnake Scraper")
	log.Printf("  Database: %s", *dbPath)
	log.Printf("  Workers: %d", *numWorkers)
	log.Printf("  Max Players: %d", *maxPlayers)
	log.Printf("  Request Delay: %s", *requestDelay)
	log.Printf("  Daemon Mode: %v", *daemon)
	log.Printf("  Auto Export: %v", *autoExport)
	log.Printf("  Output Dir: %s", *outputDir)

	// Setup signal handling for graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	// Run scraper
	if *daemon {
		runDaemon(database, *numWorkers, *maxPlayers, *requestDelay, *daemonInterval, *autoExport, *outputDir, sigChan)
	} else {
		runOnce(database, *numWorkers, *maxPlayers, *requestDelay)
		if *autoExport {
			autoExportGames(database, *outputDir)
		}
	}

	printStats(database)
}

func runOnce(database *db.DB, numWorkers, maxPlayers int, requestDelay time.Duration) {
	// Get existing game IDs for deduplication
	existingIDs, err := database.GetAllGameIDs()
	if err != nil {
		log.Printf("Warning: Failed to load existing IDs: %v", err)
		existingIDs = make(map[string]bool)
	}
	log.Printf("Loaded %d existing game IDs", len(existingIDs))

	// Create channels
	gameIDChan := make(chan string, 1000) // Buffered channel for game IDs
	done := make(chan struct{})

	// Start download workers
	dlConfig := downloader.Config{
		NumWorkers:     numWorkers,
		EngineURL:      "wss://engine.battlesnake.com/games/%s/events",
		ConnectTimeout: 10 * time.Second,
		ReadTimeout:    30 * time.Second,
	}
	dlWorker := downloader.NewWorker(dlConfig, database)
	go dlWorker.Start(gameIDChan, done)

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

	if err := discWorker.Discover(gameIDChan); err != nil {
		log.Printf("Discovery error: %v", err)
	}

	// Close the channel to signal download workers to finish
	close(gameIDChan)

	// Wait for downloads to complete
	<-done

	// Print stats
	stats := dlWorker.GetStats()
	log.Printf("Scraping complete:")
	log.Printf("  Games downloaded: %d", stats.GamesDownloaded)
	log.Printf("  Games skipped: %d", stats.GamesSkipped)
	log.Printf("  Games failed: %d", stats.GamesFailed)
	log.Printf("  Total frames: %d", stats.FramesTotal)
}

func runDaemon(database *db.DB, numWorkers, maxPlayers int, requestDelay, interval time.Duration, autoExport bool, outputDir string, sigChan chan os.Signal) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	// Run immediately first
	runOnce(database, numWorkers, maxPlayers, requestDelay)
	if autoExport {
		autoExportGames(database, outputDir)
	}

	for {
		select {
		case <-ticker.C:
			log.Printf("Starting scheduled discovery run...")
			runOnce(database, numWorkers, maxPlayers, requestDelay)
			if autoExport {
				autoExportGames(database, outputDir)
			}
		case sig := <-sigChan:
			log.Printf("Received signal %v, shutting down...", sig)
			return
		}
	}
}

func printStats(database *db.DB) {
	totalGames, processedGames, totalFrames, err := database.Stats()
	if err != nil {
		log.Printf("Failed to get stats: %v", err)
		return
	}

	fmt.Println("\n=== Database Statistics ===")
	fmt.Printf("Total games: %d\n", totalGames)
	fmt.Printf("Processed games: %d\n", processedGames)
	fmt.Printf("Unprocessed games: %d\n", totalGames-processedGames)
	fmt.Printf("Total frames: %d\n", totalFrames)
	if totalGames > 0 {
		fmt.Printf("Average frames/game: %.1f\n", float64(totalFrames)/float64(totalGames))
	}
}

func exportTrainingData(database *db.DB, maxGames int, outputPath string) error {
	log.Printf("Exporting up to %d games to %s", maxGames, outputPath)

	exp := exporter.NewExporter(database)
	trainingData, err := exp.ExportToProto(maxGames)
	if err != nil {
		return err
	}

	if len(trainingData.Examples) == 0 {
		log.Println("No examples to export (all games may already be processed)")
		return nil
	}

	// Ensure output directory exists
	if err := os.MkdirAll(filepath.Dir(outputPath), 0755); err != nil {
		return fmt.Errorf("failed to create output directory: %w", err)
	}

	data, err := proto.Marshal(trainingData)
	if err != nil {
		return fmt.Errorf("failed to marshal training data: %w", err)
	}

	if err := os.WriteFile(outputPath, data, 0644); err != nil {
		return fmt.Errorf("failed to write file: %w", err)
	}

	log.Printf("Successfully exported %d training examples", len(trainingData.Examples))
	return nil
}

// Additional utility: list games with their details
func listGames(database *db.DB, limit int) {
	games, err := database.GetUnprocessedGames(limit)
	if err != nil {
		log.Printf("Failed to get games: %v", err)
		return
	}

	fmt.Printf("\n=== Unprocessed Games (limit %d) ===\n", limit)
	for _, g := range games {
		fmt.Printf("  %s - Winner: %s, Ruleset: %s\n", g.ID, g.Winner, g.Ruleset)
	}
}

// Batch export function to match existing training data format
func exportBatch(database *db.DB, batchSize int, outputDir string) error {
	games, err := database.GetUnprocessedGames(batchSize)
	if err != nil {
		return err
	}

	exp := exporter.NewExporter(database)
	totalExamples := 0

	for _, game := range games {
		examples, err := exp.ConvertGame(game.ID)
		if err != nil {
			log.Printf("Failed to convert game %s: %v", game.ID, err)
			continue
		}

		if len(examples) == 0 {
			continue
		}

		// Create training data proto
		td := &pb.TrainingData{Examples: examples}
		data, err := proto.Marshal(td)
		if err != nil {
			log.Printf("Failed to marshal game %s: %v", game.ID, err)
			continue
		}

		// Write to file matching existing format: game_{timestamp}_{turn}.pb
		filename := fmt.Sprintf("game_scraped_%s.pb", game.ID)
		outputPath := filepath.Join(outputDir, filename)

		if err := os.WriteFile(outputPath, data, 0644); err != nil {
			log.Printf("Failed to write %s: %v", outputPath, err)
			continue
		}

		totalExamples += len(examples)

		// Mark as processed
		if err := database.MarkGameProcessed(game.ID); err != nil {
			log.Printf("Failed to mark game %s as processed: %v", game.ID, err)
		}
	}

	log.Printf("Exported %d examples from %d games to %s", totalExamples, len(games), outputDir)
	return nil
}

// autoExportGames exports all unprocessed games to individual .pb files
func autoExportGames(database *db.DB, outputDir string) {
	// Ensure output directory exists
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		log.Printf("Failed to create output directory: %v", err)
		return
	}

	// Get all unprocessed games
	games, err := database.GetUnprocessedGames(10000)
	if err != nil {
		log.Printf("Failed to get unprocessed games: %v", err)
		return
	}

	if len(games) == 0 {
		log.Printf("No unprocessed games to export")
		return
	}

	log.Printf("Auto-exporting %d games to %s", len(games), outputDir)

	exp := exporter.NewExporter(database)
	totalExamples := 0
	exportedGames := 0

	for _, game := range games {
		examples, err := exp.ConvertGame(game.ID)
		if err != nil {
			log.Printf("Failed to convert game %s: %v", game.ID, err)
			continue
		}

		if len(examples) == 0 {
			continue
		}

		// Create training data proto
		td := &pb.TrainingData{Examples: examples}
		data, err := proto.Marshal(td)
		if err != nil {
			log.Printf("Failed to marshal game %s: %v", game.ID, err)
			continue
		}

		// Write to file with timestamp for uniqueness
		filename := fmt.Sprintf("game_%d_%s.pb", time.Now().UnixNano(), game.ID[:8])
		outputPath := filepath.Join(outputDir, filename)

		if err := os.WriteFile(outputPath, data, 0644); err != nil {
			log.Printf("Failed to write %s: %v", outputPath, err)
			continue
		}

		totalExamples += len(examples)
		exportedGames++

		// Mark as processed
		if err := database.MarkGameProcessed(game.ID); err != nil {
			log.Printf("Failed to mark game %s as processed: %v", game.ID, err)
		}
	}

	log.Printf("Auto-export complete: %d games, %d examples", exportedGames, totalExamples)
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
