package discovery

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"regexp"
	"sync"
	"time"

	"github.com/PuerkitoBio/goquery"
)

// Config holds discovery worker configuration
type Config struct {
	LeaderboardURLs []string      // Multiple leaderboard URLs to scrape
	RequestDelay    time.Duration // Delay between HTTP requests to be polite
	MaxPlayers      int           // Maximum number of players to check per leaderboard (0 = unlimited)
}

// DefaultConfig returns sensible defaults
func DefaultConfig() Config {
	return Config{
		LeaderboardURLs: []string{
			"https://play.battlesnake.com/leaderboard/standard",
			"https://play.battlesnake.com/leaderboard/standard-duels",
		},
		RequestDelay: 500 * time.Millisecond,
		MaxPlayers:   0, // Unlimited by default
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

// Stats reports what a discovery pass saw.
//
// KnownGameIDs includes both:
// - duplicates across multiple players/leaderboards during discovery
// - game IDs already present in the initial known set
type Stats struct {
	Leaderboards int
	Players      int
	GameLinks    int
	NewGameIDs   int
	KnownGameIDs int
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
		// Matches /leaderboard/{arena}/{username}/stats (standard, standard-duels, etc.)
		playerRe: regexp.MustCompile(`/leaderboard/[^/]+/([^/]+)/stats`),
	}
}

// Discover starts the discovery process and sends new game IDs to the channel.
// It is cancelable via ctx.
func (w *Worker) Discover(ctx context.Context, gameIDChan chan<- string) (Stats, error) {
	stats := Stats{}
	stats.Leaderboards = len(w.config.LeaderboardURLs)

	for _, leaderboardURL := range w.config.LeaderboardURLs {
		select {
		case <-ctx.Done():
			return stats, ctx.Err()
		default:
		}
		// Get player usernames from this leaderboard
		players, arenaType, err := w.getLeaderboardPlayers(ctx, leaderboardURL)
		if err != nil {
			slog.Warn("discovery: leaderboard fetch failed", "leaderboard_url", leaderboardURL, "error", err)
			continue
		}
		_ = arenaType
		stats.Players += len(players)

		// Limit players if configured
		if w.config.MaxPlayers > 0 && len(players) > w.config.MaxPlayers {
			players = players[:w.config.MaxPlayers]
		}

		// Crawl each player's game history
		for i, player := range players {
			_ = i
			select {
			case <-ctx.Done():
				return stats, ctx.Err()
			default:
			}
			gameIDs, err := w.getPlayerGames(ctx, player.statsURL)
			if err != nil {
				slog.Warn("discovery: player games fetch failed", "username", player.username, "error", err)
				continue
			}
			stats.GameLinks += len(gameIDs)

			for _, gameID := range gameIDs {
				w.knownMu.RLock()
				known := w.knownIDs[gameID]
				w.knownMu.RUnlock()

				if !known {
					w.knownMu.Lock()
					w.knownIDs[gameID] = true
					w.knownMu.Unlock()

					select {
					case gameIDChan <- gameID:
						stats.NewGameIDs++
					case <-ctx.Done():
						return stats, ctx.Err()
					}
				} else {
					stats.KnownGameIDs++
				}
			}

			// Rate limiting
			if w.config.RequestDelay > 0 {
				t := time.NewTimer(w.config.RequestDelay)
				select {
				case <-ctx.Done():
					t.Stop()
					return stats, ctx.Err()
				case <-t.C:
				}
			}
		}
	}

	return stats, nil
}

// playerInfo holds player data from a leaderboard
type playerInfo struct {
	username string
	statsURL string
}

// getLeaderboardPlayers fetches player usernames from the leaderboard page
func (w *Worker) getLeaderboardPlayers(ctx context.Context, leaderboardURL string) ([]playerInfo, string, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", leaderboardURL, nil)
	if err != nil {
		return nil, "", err
	}
	req.Header.Set("User-Agent", "BattlesnakeScraper/1.0 (training-data-collector)")

	resp, err := w.client.Do(req)
	if err != nil {
		return nil, "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return nil, "", fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	doc, err := goquery.NewDocumentFromReader(resp.Body)
	if err != nil {
		return nil, "", err
	}

	// Extract arena type from URL (e.g., "standard" or "standard-duels")
	arenaType := "unknown"
	arenaRe := regexp.MustCompile(`/leaderboard/([^/]+)/?$`)
	if matches := arenaRe.FindStringSubmatch(leaderboardURL); len(matches) >= 2 {
		arenaType = matches[1]
	}

	// Find all player links - pattern is /leaderboard/{arena}/{username}/stats
	var players []playerInfo
	seen := make(map[string]bool)

	doc.Find("a[href*='/leaderboard/']").Each(func(i int, s *goquery.Selection) {
		href, exists := s.Attr("href")
		if !exists {
			return
		}

		matches := w.playerRe.FindStringSubmatch(href)
		if len(matches) >= 2 {
			username := matches[1]
			if !seen[username] {
				seen[username] = true
				// Build the full stats URL
				statsURL := "https://play.battlesnake.com" + href
				players = append(players, playerInfo{
					username: username,
					statsURL: statsURL,
				})
			}
		}
	})

	return players, arenaType, nil
}

// getPlayerGames fetches game IDs from a player's stats page
func (w *Worker) getPlayerGames(ctx context.Context, statsURL string) ([]string, error) {
	var gameIDs []string
	seen := make(map[string]bool)

	req, err := http.NewRequestWithContext(ctx, "GET", statsURL, nil)
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
