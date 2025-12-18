package rules

import (
	pb "github.com/brensch/snek2/gen/go"
	"google.golang.org/protobuf/proto"
)

const (
	MoveUp    = 0
	MoveDown  = 1
	MoveLeft  = 2
	MoveRight = 3
)

// GetLegalMoves returns a list of legal moves for the snake identified by YouId
func GetLegalMoves(state *pb.GameState) []int {
	var you *pb.Snake
	for _, s := range state.Snakes {
		if s.Id == state.YouId {
			you = s
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
		p    *pb.Point
	}{
		{MoveUp, &pb.Point{X: head.X, Y: head.Y + 1}},
		{MoveDown, &pb.Point{X: head.X, Y: head.Y - 1}},
		{MoveLeft, &pb.Point{X: head.X - 1, Y: head.Y}},
		{MoveRight, &pb.Point{X: head.X + 1, Y: head.Y}},
	}

	for _, c := range candidates {
		if isSafe(state, c.p, you.Body) {
			moves = append(moves, c.move)
		}
	}

	return moves
}

func isSafe(state *pb.GameState, p *pb.Point, myBody []*pb.Point) bool {
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
func NextState(state *pb.GameState, move int) *pb.GameState {
	// Deep copy state
	newState := proto.Clone(state).(*pb.GameState)
	newState.Turn++

	var you *pb.Snake
	for _, s := range newState.Snakes {
		if s.Id == newState.YouId {
			you = s
			break
		}
	}

	if you == nil || you.Health <= 0 {
		return newState
	}

	// Calculate new head
	head := you.Body[0]
	newHead := &pb.Point{X: head.X, Y: head.Y}

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
	newBody := []*pb.Point{newHead}
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

// IsTerminal returns true if the game is over for You
func IsTerminal(state *pb.GameState) bool {
	var you *pb.Snake
	for _, s := range state.Snakes {
		if s.Id == state.YouId {
			you = s
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
func GetResult(state *pb.GameState) float32 {
	if IsTerminal(state) {
		// If we are terminal, we lost (unless we are the last one standing, but let's assume simple survival)
		return -1.0
	}
	return 0.0
}
