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

	// ModelPath is the resolved path to the ONNX model used for selfplay games.
	ModelPath string `json:"model_path,omitempty"`

	// MCTSRoot contains the summary of MCTS root children for unified search.
	// This is present in selfplay games and can be used to show the decision.
	MCTSRoot []JointChildSummary `json:"mcts_root,omitempty"`
}

// JointChildSummary is a compact representation of a joint action explored at the root.
type JointChildSummary struct {
	Moves      map[string]int `json:"moves"`     // snakeID -> move
	VisitCount int            `json:"n"`         // Visit count
	ValueSum   float32        `json:"value_sum"` // Sum of values
	Q          float32        `json:"q"`         // Average value
	PriorProb  float32        `json:"p"`         // Joint prior
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

// DebugMCTSNode is a node in the MCTS tree for debug visualization.
// Alternating format: Each node represents ONE snake's move decision.
type DebugMCTSNode struct {
	// Moves is the action - for alternating tree, it's just {snakeID: move}
	// nil for root node
	Moves map[string]int `json:"moves,omitempty"`

	VisitCount int     `json:"n"`
	ValueSum   float32 `json:"value_sum"`
	Q          float32 `json:"q"`
	PriorProb  float32 `json:"p"`
	UCB        float32 `json:"ucb"`

	// SnakeID is the snake whose turn it is at this node
	SnakeID string `json:"snake_id,omitempty"`

	// SnakeIndex is the position in the snake order (0, 1, 2, ...)
	SnakeIndex int `json:"snake_index"`

	// State is the game state (only set at the start of each round when all snakes have moved)
	State *DebugGameState `json:"state,omitempty"`

	// SnakePriors shows each snake's policy at this node
	SnakePriors map[string][4]float32 `json:"snake_priors,omitempty"`

	Children []*DebugMCTSNode `json:"children,omitempty"`

	// Legacy fields for old format compatibility
	Move         int            `json:"move,omitempty"`
	IsResolved   bool           `json:"is_resolved,omitempty"`
	PendingMoves map[string]int `json:"pending_moves,omitempty"`
	NextSnake    string         `json:"next_snake,omitempty"`
}

// DebugGameState is a minimal game state for debug visualization.
type DebugGameState struct {
	Turn   int32             `json:"turn"`
	Width  int32             `json:"width"`
	Height int32             `json:"height"`
	YouId  string            `json:"you_id"`
	Food   []Point           `json:"food"`
	Snakes []DebugSnakeState `json:"snakes"`
}

// DebugSnakeState is a minimal snake state for debug visualization.
type DebugSnakeState struct {
	ID     string  `json:"id"`
	Health int32   `json:"health"`
	Body   []Point `json:"body"`
	Alive  bool    `json:"alive"`
}

// DebugSnakeTree holds the MCTS tree for one snake at a turn.
// DEPRECATED: V2 uses a single shared tree instead.
type DebugSnakeTree struct {
	SnakeID  string         `json:"snake_id"`
	SnakeIdx int            `json:"snake_idx"`
	BestMove int            `json:"best_move"`
	Root     *DebugMCTSNode `json:"root"`
}

// DebugTurnData holds all MCTS trees and state for a single turn.
type DebugTurnData struct {
	GameID    string          `json:"game_id"`
	ModelPath string          `json:"model_path"`
	Turn      int32           `json:"turn"`
	Sims      int             `json:"sims"`
	Cpuct     float32         `json:"cpuct"`
	State     *DebugGameState `json:"state"`

	// V1: Separate tree per snake (deprecated)
	Trees []*DebugSnakeTree `json:"trees,omitempty"`

	// V2: Single shared tree that alternates between snakes
	SnakeOrder []string       `json:"snake_order,omitempty"`
	Tree       *DebugMCTSNode `json:"tree,omitempty"`
	ChosenPath []int          `json:"chosen_path,omitempty"`
}

// DebugGameSummary is a summary of a debug game for listing.
type DebugGameSummary struct {
	GameID    string  `json:"game_id"`
	ModelPath string  `json:"model_path"`
	TurnCount int     `json:"turn_count"`
	Sims      int     `json:"sims"`
	Cpuct     float32 `json:"cpuct"`
	FileName  string  `json:"file_name"`
}

// DebugGameResponse is the full debug game data.
type DebugGameResponse struct {
	GameID    string          `json:"game_id"`
	ModelPath string          `json:"model_path"`
	TurnCount int             `json:"turn_count"`
	Sims      int             `json:"sims"`
	Cpuct     float32         `json:"cpuct"`
	Turns     []DebugTurnData `json:"turns"`
}
