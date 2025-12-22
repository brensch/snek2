package rules

import (
	"github.com/brensch/snek2/game"
)

const (
	MoveUp    = 0
	MoveDown  = 1
	MoveLeft  = 2
	MoveRight = 3
)

// GetLegalMoves returns a list of legal moves for the snake identified by YouId
func GetLegalMoves(state *game.GameState) []int {
	var you *game.Snake
	for i := range state.Snakes {
		if state.Snakes[i].Id == state.YouId {
			you = &state.Snakes[i]
			break
		}
	}

	if you == nil || you.Health <= 0 {
		return []int{}
	}

	head := you.Body[0]
	moves := []int{}

	// Potential next positions
	candidates := []struct {
		move int
		p    game.Point
	}{
		{MoveUp, game.Point{X: head.X, Y: head.Y + 1}},
		{MoveDown, game.Point{X: head.X, Y: head.Y - 1}},
		{MoveLeft, game.Point{X: head.X - 1, Y: head.Y}},
		{MoveRight, game.Point{X: head.X + 1, Y: head.Y}},
	}

	for _, c := range candidates {
		if isSafe(state, c.p, you.Body) {
			moves = append(moves, c.move)
		}
	}

	return moves
}

func isSafe(state *game.GameState, p game.Point, myBody []game.Point) bool {
	// 1. Check Bounds
	if p.X < 0 || p.X >= state.Width || p.Y < 0 || p.Y >= state.Height {
		return false
	}

	// 2. Check Collisions with Snakes
	for _, s := range state.Snakes {
		// Skip tail if it's going to move (health < 100 implies not eating, but we don't know if they eat yet)
		// Conservative: Don't hit any body part including tail.
		for _, bp := range s.Body {
			if p.X == bp.X && p.Y == bp.Y {
				return false
			}
		}
	}

	// 3. Neck check (don't move backwards into own neck)
	// This is implicitly handled by body collision, but good to be explicit if body logic changes
	if len(myBody) > 1 {
		neck := myBody[1]
		if p.X == neck.X && p.Y == neck.Y {
			return false
		}
	}

	return true
}

// NextState returns the next state after applying a move for YouId
// Note: This currently only moves "You". Enemies are static.
// This is used primarily for MCTS expansion where we might not know enemy moves.
func NextState(state *game.GameState, move int) *game.GameState {
	newState := state.Clone()
	newState.Turn++

	var you *game.Snake
	for i := range newState.Snakes {
		if newState.Snakes[i].Id == newState.YouId {
			you = &newState.Snakes[i]
			break
		}
	}

	if you == nil || you.Health <= 0 {
		return newState
	}

	// Calculate new head
	head := you.Body[0]
	newHead := game.Point{X: head.X, Y: head.Y}

	switch move {
	case MoveUp:
		newHead.Y++
	case MoveDown:
		newHead.Y--
	case MoveLeft:
		newHead.X--
	case MoveRight:
		newHead.X++
	}

	// Check for food
	ateFood := false
	for i, f := range newState.Food {
		if f.X == newHead.X && f.Y == newHead.Y {
			ateFood = true
			// Remove food
			newState.Food = append(newState.Food[:i], newState.Food[i+1:]...)
			break
		}
	}

	// Update Body
	newBody := []game.Point{newHead}
	newBody = append(newBody, you.Body...)

	if ateFood {
		you.Health = 100
	} else {
		you.Health--
		// Remove tail
		if len(newBody) > 0 {
			newBody = newBody[:len(newBody)-1]
		}
	}
	you.Body = newBody

	return newState
}

// NextStateSimultaneous advances the game state with moves for all snakes.
func NextStateSimultaneous(state *game.GameState, moves map[string]int) *game.GameState {
	newState := state.Clone()
	newState.Turn++

	// 1. Calculate potential new heads
	newHeads := make(map[string]game.Point)
	for i := range newState.Snakes {
		s := &newState.Snakes[i]
		if s.Health <= 0 {
			continue
		}
		move, ok := moves[s.Id]
		if !ok {
			// If no move provided, snake dies (or stays still? let's say it dies/stops)
			// For now, let's assume it moves Up if missing (or just dies)
			// Better: treat as dying.
			continue
		}

		head := s.Body[0]
		newHead := game.Point{X: head.X, Y: head.Y}
		switch move {
		case MoveUp:
			newHead.Y++
		case MoveDown:
			newHead.Y--
		case MoveLeft:
			newHead.X--
		case MoveRight:
			newHead.X++
		}
		newHeads[s.Id] = newHead
	}

	// 2. Move bodies (tentatively)
	// We need to track who ate food to know if they grow (tail stays)
	eatenFood := make(map[int]bool) // index of food
	snakeAte := make(map[string]bool)

	for id, head := range newHeads {
		for i, f := range newState.Food {
			if f.X == head.X && f.Y == head.Y {
				eatenFood[i] = true
				snakeAte[id] = true
			}
		}
	}

	// Remove eaten food
	remainingFood := []game.Point{}
	for i, f := range newState.Food {
		if !eatenFood[i] {
			remainingFood = append(remainingFood, f)
		}
	}
	newState.Food = remainingFood

	// Update bodies
	for i := range newState.Snakes {
		s := &newState.Snakes[i]
		newHead, ok := newHeads[s.Id]
		if !ok {
			s.Health = 0 // No move = dead
			continue
		}

		newBody := []game.Point{newHead}
		newBody = append(newBody, s.Body...)

		if snakeAte[s.Id] {
			s.Health = 100
		} else {
			s.Health--
			if len(newBody) > 0 {
				newBody = newBody[:len(newBody)-1]
			}
		}
		s.Body = newBody
	}

	// 3. Check Collisions (Death Logic)
	// We need to mark snakes as dead.
	deadSnakes := make(map[string]bool)

	for _, s := range newState.Snakes {
		if s.Health <= 0 {
			deadSnakes[s.Id] = true
			continue
		}
		head := s.Body[0]

		// Wall Collision
		if head.X < 0 || head.X >= newState.Width || head.Y < 0 || head.Y >= newState.Height {
			deadSnakes[s.Id] = true
			continue
		}

		// Body Collision (Self and Others)
		for _, other := range newState.Snakes {
			if other.Health <= 0 {
				continue
			} // Don't collide with already dead snakes?
			// Actually, in Battlesnake, you collide with the body they *had* or *have*?
			// Standard: You collide with the body segments that exist AFTER the move.

			for i, p := range other.Body {
				if i == 0 && s.Id == other.Id {
					continue
				} // Skip own head
				if i == 0 && s.Id != other.Id {
					// Head-to-Head handled later
					continue
				}
				if p.X == head.X && p.Y == head.Y {
					deadSnakes[s.Id] = true
				}
			}
		}
	}

	// Head-to-Head Collision
	for i := 0; i < len(newState.Snakes); i++ {
		s1 := newState.Snakes[i]
		if deadSnakes[s1.Id] {
			continue
		}

		for j := i + 1; j < len(newState.Snakes); j++ {
			s2 := newState.Snakes[j]
			if deadSnakes[s2.Id] {
				continue
			}

			if s1.Body[0].X == s2.Body[0].X && s1.Body[0].Y == s2.Body[0].Y {
				// Collision!
				if len(s1.Body) > len(s2.Body) {
					deadSnakes[s2.Id] = true
				} else if len(s2.Body) > len(s1.Body) {
					deadSnakes[s1.Id] = true
				} else {
					deadSnakes[s1.Id] = true
					deadSnakes[s2.Id] = true
				}
			}
		}
	}

	// Apply Death
	finalSnakes := make([]game.Snake, 0, len(newState.Snakes))
	for _, s := range newState.Snakes {
		if deadSnakes[s.Id] {
			// Drop dead snakes from the active list.
			continue
		}
		finalSnakes = append(finalSnakes, s)
	}
	newState.Snakes = finalSnakes

	return newState
}

// IsGameOver returns true if the game is over (0 or 1 snake left)
func IsGameOver(state *game.GameState) bool {
	living := 0
	for _, s := range state.Snakes {
		if s.Health > 0 {
			living++
		}
	}
	return living <= 1
}

// IsTerminal returns true if the game is over for You
func IsTerminal(state *game.GameState) bool {
	var you *game.Snake
	for i := range state.Snakes {
		if state.Snakes[i].Id == state.YouId {
			you = &state.Snakes[i]
			break
		}
	}

	if you == nil {
		return true
	}

	if you.Health <= 0 {
		return true
	}

	// Check if head is out of bounds or colliding
	// Note: In MCTS, we usually check IsTerminal on a node that was just created.
	// If we created a node with an illegal move, it shouldn't exist.
	// But if we are in a state where we have NO legal moves, we are effectively dead.
	if len(GetLegalMoves(state)) == 0 {
		return true
	}

	return false
}

// GetResult returns the result of the game (+1 for win, -1 for loss)
func GetResult(state *game.GameState) float32 {
	if IsTerminal(state) {
		// If we are terminal, we lost (unless we are the last one standing, but let's assume simple survival)
		return -1.0
	}
	return 0.0
}
