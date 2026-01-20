// Package game defines the core game state types for Battlesnake.
//
// These types represent the minimal state needed for rules evaluation and
// neural network inference. The state is designed to be efficiently clonable
// for MCTS tree exploration.
package game

// Point is a board coordinate.
// Coordinates follow Battlesnake conventions: (0,0) is bottom-left.
type Point struct {
	X int32
	Y int32
}

type Snake struct {
	Id     string
	Health int32
	Body   []Point
}

// GameState is the complete state needed for rules + inference.
// YouId selects the ego snake perspective for inference/encoding.
type GameState struct {
	Width  int32
	Height int32
	Snakes []Snake
	Food   []Point
	YouId  string
	Turn   int32
}

// Clone performs a deep copy of the game state.
func (s *GameState) Clone() *GameState {
	if s == nil {
		return nil
	}

	out := &GameState{
		Width:  s.Width,
		Height: s.Height,
		YouId:  s.YouId,
		Turn:   s.Turn,
	}

	if len(s.Food) > 0 {
		out.Food = make([]Point, len(s.Food))
		copy(out.Food, s.Food)
	}

	if len(s.Snakes) > 0 {
		out.Snakes = make([]Snake, len(s.Snakes))
		for i := range s.Snakes {
			out.Snakes[i] = Snake{Id: s.Snakes[i].Id, Health: s.Snakes[i].Health}
			if len(s.Snakes[i].Body) > 0 {
				out.Snakes[i].Body = make([]Point, len(s.Snakes[i].Body))
				copy(out.Snakes[i].Body, s.Snakes[i].Body)
			}
		}
	}

	return out
}
