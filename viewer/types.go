package main

// GameSummary represents a summarized game entry for the games list.
type GameSummary struct {
	GameID string `json:"game_id"`
	// StartedNs is parsed from game_id for selfplay games with format: selfplay_<unix_nano>_<worker>.
	// Nil for game IDs that do not match that format.
	StartedNs  *int64 `json:"started_ns"`
	MinTurn    int32  `json:"min_turn"`
	MaxTurn    int32  `json:"max_turn"`
	TurnCount  int32  `json:"turn_count"`
	Width      int32  `json:"width"`
	Height     int32  `json:"height"`
	Source     string `json:"source"`
	SourceFile string `json:"file"`
	Results    string `json:"results"`
}

// GamesResponse is the paginated response for the /api/games endpoint.
type GamesResponse struct {
	Total int64         `json:"total"`
	Games []GameSummary `json:"games"`
}

// StatsPoint represents a single time-bucket in the stats response.
type StatsPoint struct {
	TNs int64 `json:"t_ns"`

	SelfplayGames      int64 `json:"selfplay_games"`
	SelfplayTotalTurns int64 `json:"selfplay_total_turns"`
	SelfplayWins       int64 `json:"selfplay_wins"`
	SelfplayDraws      int64 `json:"selfplay_draws"`

	ScrapedGames      int64 `json:"scraped_games"`
	ScrapedTotalTurns int64 `json:"scraped_total_turns"`
	ScrapedWins       int64 `json:"scraped_wins"`
	ScrapedDraws      int64 `json:"scraped_draws"`
}

// StatsResponse is the response for the /api/stats endpoint.
type StatsResponse struct {
	FromNs   int64        `json:"from_ns"`
	ToNs     int64        `json:"to_ns"`
	BucketNs int64        `json:"bucket_ns"`
	Points   []StatsPoint `json:"points"`
}

// Point represents a 2D coordinate on the board.
type Point struct {
	X int32 `json:"x"`
	Y int32 `json:"y"`
}

// Snake represents a snake in a turn.
type Snake struct {
	ID          string    `json:"id"`
	Alive       bool      `json:"alive"`
	Health      int32     `json:"health"`
	Body        []Point   `json:"body"`
	Policy      int32     `json:"policy"`
	PolicyProbs []float32 `json:"policy_probs,omitempty"`
	Value       float32   `json:"value"`
}

// Turn represents a single turn in a game.
type Turn struct {
	GameID  string  `json:"game_id"`
	Turn    int32   `json:"turn"`
	Width   int32   `json:"width"`
	Height  int32   `json:"height"`
	Food    []Point `json:"food"`
	Hazards []Point `json:"hazards"`
	Snakes  []Snake `json:"snakes"`
	Source  string  `json:"source"`
}

// MCTSMove represents a possible move in the MCTS tree.
type MCTSMove struct {
	Move   int       `json:"move"`
	Exists bool      `json:"exists"`
	N      int       `json:"n"`
	Q      float32   `json:"q"`
	P      float32   `json:"p"`
	UCB    float32   `json:"ucb"`
	Child  *MCTSNode `json:"child,omitempty"`
}

// MCTSNode represents a node in the MCTS search tree.
type MCTSNode struct {
	VisitCount int         `json:"visit_count"`
	Value      float32     `json:"value"`
	State      *Turn       `json:"state,omitempty"`
	Moves      [4]MCTSMove `json:"moves"`
}

// MCTSResponse is the response for the /api/mcts endpoint.
type MCTSResponse struct {
	GameID   string    `json:"game_id"`
	Turn     int32     `json:"turn"`
	EgoIdx   int       `json:"ego_idx"`
	EgoID    string    `json:"ego_id"`
	Sims     int       `json:"sims"`
	Cpuct    float32   `json:"cpuct"`
	Depth    int       `json:"depth"`
	MaxDepth int       `json:"max_depth"`
	BestMove int       `json:"best_move"`
	Root     *MCTSNode `json:"root"`
}

// MCTSSnakeTree represents MCTS results for a single snake.
type MCTSSnakeTree struct {
	SnakeIdx int       `json:"snake_idx"`
	SnakeID  string    `json:"snake_id"`
	BestMove int       `json:"best_move"`
	Root     *MCTSNode `json:"root"`
}

// MCTSAllResponse emulates the selfplay generation logic:
// run an independent MCTS for each living snake (each with its own YouId perspective)
// on the same position.
type MCTSAllResponse struct {
	GameID string          `json:"game_id"`
	Turn   int32           `json:"turn"`
	Sims   int             `json:"sims"`
	Cpuct  float32         `json:"cpuct"`
	Depth  int             `json:"depth"`
	State  Turn            `json:"state"`
	Snakes []MCTSSnakeTree `json:"snakes"`
}

// SimulateRequest is the request body for the /api/simulate endpoint.
type SimulateRequest struct {
	State Turn           `json:"state"`
	Moves map[string]int `json:"moves"`
}
