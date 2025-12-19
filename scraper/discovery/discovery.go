package discovery

import (
	"fmt"
	"log"
	"net/http"
	"regexp"
	"sync"
	"time"

	"github.com/PuerkitoBio/goquery"
)

// Config holds discovery worker configuration
type Config struct {
	LeaderboardURL string
	RequestDelay   time.Duration // Delay between HTTP requests to be polite
	MaxPlayers     int           // Maximum number of players to check (0 = unlimited)
}

// DefaultConfig returns sensible defaults
func DefaultConfig() Config {
	return Config{
		LeaderboardURL: "https://play.battlesnake.com/leaderboard/standard",
		RequestDelay:   500 * time.Millisecond,
		MaxPlayers:     100, // Check top 100 players by default
	}
}

// Worker discovers game IDs from the Battlesnake leaderboard
type Worker struct {
	config   Config
	client   *http.Client
	knownIDs map[string]bool
	knownMu  sync.RWMutex
	gameIDRe *regexp.Regexp
	playerRe *regexp.Regexp
}

// NewWorker creates a new discovery worker
func NewWorker(config Config, existingIDs map[string]bool) *Worker {
	if existingIDs == nil {
		existingIDs = make(map[string]bool)
	}

	return &Worker{
		config: config,
		client: &http.Client{
			Timeout: 30 * time.Second,
		},
		knownIDs: existingIDs,
		gameIDRe: regexp.MustCompile(`/game/([a-f0-9-]+)`),
		// Matches /leaderboard/standard/{username}/stats
		playerRe: regexp.MustCompile(`/leaderboard/standard/([^/]+)/stats`),
	}
}

// Discover starts the discovery process and sends new game IDs to the channel
func (w *Worker) Discover(gameIDChan chan<- string) error {
	log.Println("[Discovery] Starting leaderboard crawl...")

	// Get player usernames from leaderboard
	players, err := w.getLeaderboardPlayers()
	if err != nil {
		return fmt.Errorf("failed to get leaderboard: %w", err)
	}

	log.Printf("[Discovery] Found %d players on leaderboard", len(players))

	// Limit players if configured
	if w.config.MaxPlayers > 0 && len(players) > w.config.MaxPlayers {
		players = players[:w.config.MaxPlayers]
	}

	// Crawl each player's game history
	newGames := 0
	for i, player := range players {
		log.Printf("[Discovery] Checking player %d/%d: %s", i+1, len(players), player)

		gameIDs, err := w.getPlayerGames(player)
		if err != nil {
			log.Printf("[Discovery] Error getting games for %s: %v", player, err)
			continue
		}

		for _, gameID := range gameIDs {
			w.knownMu.RLock()
			known := w.knownIDs[gameID]
			w.knownMu.RUnlock()

			if !known {
				w.knownMu.Lock()
				w.knownIDs[gameID] = true
				w.knownMu.Unlock()

				gameIDChan <- gameID
				newGames++
			}
		}

		// Rate limiting
		time.Sleep(w.config.RequestDelay)
	}

	log.Printf("[Discovery] Finished. Found %d new games", newGames)
	return nil
}

// getLeaderboardPlayers fetches player usernames from the leaderboard page
func (w *Worker) getLeaderboardPlayers() ([]string, error) {
	req, err := http.NewRequest("GET", w.config.LeaderboardURL, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", "BattlesnakeScraper/1.0 (training-data-collector)")

	resp, err := w.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	doc, err := goquery.NewDocumentFromReader(resp.Body)
	if err != nil {
		return nil, err
	}

	// Find all player links - pattern is /leaderboard/standard/{username}/stats
	var players []string
	seen := make(map[string]bool)

	doc.Find("a[href*='/leaderboard/standard/']").Each(func(i int, s *goquery.Selection) {
		href, exists := s.Attr("href")
		if !exists {
			return
		}

		matches := w.playerRe.FindStringSubmatch(href)
		if len(matches) >= 2 {
			username := matches[1]
			if !seen[username] {
				seen[username] = true
				players = append(players, username)
			}
		}
	})

	return players, nil
}

// getPlayerGames fetches game IDs from a player's stats page
func (w *Worker) getPlayerGames(username string) ([]string, error) {
	// Stats page URL pattern from leaderboard
	url := fmt.Sprintf("https://play.battlesnake.com/leaderboard/standard/%s/stats", username)

	var gameIDs []string
	seen := make(map[string]bool)

	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", "BattlesnakeScraper/1.0 (training-data-collector)")

	resp, err := w.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("status %d", resp.StatusCode)
	}

	doc, err := goquery.NewDocumentFromReader(resp.Body)
	if err != nil {
		return nil, err
	}

	// Find all game links
	doc.Find("a[href*='/game/']").Each(func(i int, s *goquery.Selection) {
		href, exists := s.Attr("href")
		if !exists {
			return
		}

		matches := w.gameIDRe.FindStringSubmatch(href)
		if len(matches) >= 2 {
			gameID := matches[1]
			if !seen[gameID] {
				seen[gameID] = true
				gameIDs = append(gameIDs, gameID)
			}
		}
	})

	return gameIDs, nil
}

// AddKnownID adds a game ID to the known set (used for deduplication)
func (w *Worker) AddKnownID(gameID string) {
	w.knownMu.Lock()
	defer w.knownMu.Unlock()
	w.knownIDs[gameID] = true
}
