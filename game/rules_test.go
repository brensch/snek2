package game

import (
	"strings"
	"testing"
)

// dumpState is a test helper to visualize board state.
func dumpState(state *GameState) string {
	grid := make([][]byte, state.Height)
	for y := int32(0); y < state.Height; y++ {
		grid[y] = make([]byte, state.Width)
		for x := int32(0); x < state.Width; x++ {
			grid[y][x] = '.'
		}
	}
	for _, f := range state.Food {
		if f.Y >= 0 && f.Y < state.Height && f.X >= 0 && f.X < state.Width {
			grid[f.Y][f.X] = '*'
		}
	}
	for i, s := range state.Snakes {
		sym := byte('a' + i)
		for j, p := range s.Body {
			if p.Y >= 0 && p.Y < state.Height && p.X >= 0 && p.X < state.Width {
				if j == 0 {
					grid[p.Y][p.X] = sym - 32 // uppercase head
				} else {
					grid[p.Y][p.X] = sym
				}
			}
		}
	}
	var sb strings.Builder
	for y := state.Height - 1; y >= 0; y-- {
		sb.WriteString(string(grid[y]))
		sb.WriteByte('\n')
	}
	return sb.String()
}

func logNextState(t *testing.T, label string, before *GameState, move int, after *GameState) {
	moveNames := []string{"up", "down", "left", "right"}
	t.Logf("%s\n  BEFORE (move=%s):\n%s  AFTER:\n%s", label, moveNames[move], dumpState(before), dumpState(after))
}

func logNextStateSimultaneous(t *testing.T, label string, before *GameState, moves map[string]int, after *GameState) {
	moveNames := []string{"up", "down", "left", "right"}
	var moveStrs []string
	for id, m := range moves {
		moveStrs = append(moveStrs, id+"="+moveNames[m])
	}
	t.Logf("%s\n  BEFORE (moves=%v):\n%s  AFTER:\n%s", label, moveStrs, dumpState(before), dumpState(after))
}

func TestNextState_NormalMove_NoFood(t *testing.T) {
	before := &GameState{
		Width:  7,
		Height: 7,
		YouId:  "me",
		Snakes: []Snake{{
			Id:     "me",
			Health: 100,
			Body:   []Point{{X: 3, Y: 3}, {X: 3, Y: 2}, {X: 3, Y: 1}},
		}},
		Food: []Point{},
		Turn: 0,
	}

	after := NextStateWithFoodSettings(before, MoveUp, nil, FoodSettings{MinimumFood: 0, FoodSpawnChance: 0})
	logNextState(t, "NextState normal move (no food)", before, MoveUp, after)

	got := after.Snakes[0].Body
	want := []Point{{X: 3, Y: 4}, {X: 3, Y: 3}, {X: 3, Y: 2}}
	if len(got) != len(want) {
		t.Fatalf("body len=%d want=%d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("body[%d]=%v want=%v", i, got[i], want[i])
		}
	}
	if after.Snakes[0].Health != 99 {
		t.Fatalf("health=%d want=99", after.Snakes[0].Health)
	}
}

func TestNextState_EatFood_GrowsByAppendingTail(t *testing.T) {
	before := &GameState{
		Width:  7,
		Height: 7,
		YouId:  "me",
		Snakes: []Snake{{
			Id:     "me",
			Health: 50,
			Body:   []Point{{X: 3, Y: 3}, {X: 3, Y: 2}, {X: 3, Y: 1}},
		}},
		Food: []Point{{X: 3, Y: 4}},
		Turn: 0,
	}

	after := NextStateWithFoodSettings(before, MoveUp, nil, FoodSettings{MinimumFood: 0, FoodSpawnChance: 0})
	logNextState(t, "NextState eat food", before, MoveUp, after)

	got := after.Snakes[0].Body
	want := []Point{{X: 3, Y: 4}, {X: 3, Y: 3}, {X: 3, Y: 2}, {X: 3, Y: 2}}
	if len(got) != len(want) {
		t.Fatalf("body len=%d want=%d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("body[%d]=%v want=%v", i, got[i], want[i])
		}
	}
	if after.Snakes[0].Health != 100 {
		t.Fatalf("health=%d want=100", after.Snakes[0].Health)
	}
	if len(after.Food) != 0 {
		t.Fatalf("food len=%d want=0", len(after.Food))
	}
}

func TestNextState_StackedSpawn_EatFood(t *testing.T) {
	before := &GameState{
		Width:  7,
		Height: 7,
		YouId:  "me",
		Snakes: []Snake{{
			Id:     "me",
			Health: 10,
			Body:   []Point{{X: 1, Y: 1}, {X: 1, Y: 1}, {X: 1, Y: 1}},
		}},
		Food: []Point{{X: 1, Y: 2}},
		Turn: 0,
	}

	after := NextStateWithFoodSettings(before, MoveUp, nil, FoodSettings{MinimumFood: 0, FoodSpawnChance: 0})
	logNextState(t, "NextState stacked spawn eat", before, MoveUp, after)

	got := after.Snakes[0].Body
	want := []Point{{X: 1, Y: 2}, {X: 1, Y: 1}, {X: 1, Y: 1}, {X: 1, Y: 1}}
	if len(got) != len(want) {
		t.Fatalf("body len=%d want=%d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("body[%d]=%v want=%v", i, got[i], want[i])
		}
	}
}

func TestNextStateSimultaneous_BothMove_OneEats(t *testing.T) {
	before := &GameState{
		Width:  7,
		Height: 7,
		YouId:  "a",
		Snakes: []Snake{
			{Id: "a", Health: 10, Body: []Point{{X: 1, Y: 1}, {X: 1, Y: 1}, {X: 1, Y: 1}}},
			{Id: "b", Health: 10, Body: []Point{{X: 5, Y: 5}, {X: 5, Y: 5}, {X: 5, Y: 5}}},
		},
		Food: []Point{{X: 1, Y: 2}},
		Turn: 0,
	}

	moves := map[string]int{"a": MoveUp, "b": MoveLeft}
	after := NextStateSimultaneousWithFoodSettings(before, moves, nil, FoodSettings{MinimumFood: 0, FoodSpawnChance: 0})
	logNextStateSimultaneous(t, "NextStateSimultaneous one eats", before, moves, after)

	// Snake a ate: should grow by duplicating new tail.
	var a, b *Snake
	for i := range after.Snakes {
		s := &after.Snakes[i]
		if s.Id == "a" {
			a = s
		}
		if s.Id == "b" {
			b = s
		}
	}
	if a == nil || b == nil {
		t.Fatalf("expected both snakes alive")
	}
	if len(a.Body) != 4 {
		t.Fatalf("snake a len=%d want=4", len(a.Body))
	}
	wantA := []Point{{X: 1, Y: 2}, {X: 1, Y: 1}, {X: 1, Y: 1}, {X: 1, Y: 1}}
	for i := range wantA {
		if a.Body[i] != wantA[i] {
			t.Fatalf("snake a body[%d]=%v want=%v", i, a.Body[i], wantA[i])
		}
	}
	if a.Health != 100 {
		t.Fatalf("snake a health=%d want=100", a.Health)
	}

	// Snake b did not eat: normal move keeps len 3.
	if len(b.Body) != 3 {
		t.Fatalf("snake b len=%d want=3", len(b.Body))
	}
	wantB := []Point{{X: 4, Y: 5}, {X: 5, Y: 5}, {X: 5, Y: 5}}
	for i := range wantB {
		if b.Body[i] != wantB[i] {
			t.Fatalf("snake b body[%d]=%v want=%v", i, b.Body[i], wantB[i])
		}
	}
	if b.Health != 9 {
		t.Fatalf("snake b health=%d want=9", b.Health)
	}

	if len(after.Food) != 0 {
		t.Fatalf("food len=%d want=0", len(after.Food))
	}
}

func TestFood_MinimumFoodIsEnforced(t *testing.T) {
	before := &GameState{
		Width:  5,
		Height: 5,
		YouId:  "me",
		Snakes: []Snake{{Id: "me", Health: 100, Body: []Point{{X: 2, Y: 2}, {X: 2, Y: 2}, {X: 2, Y: 2}}}},
		Food:   nil,
		Turn:   0,
	}

	// Use deterministic behavior by leaving rng=nil, but force minimum food.
	after := NextStateWithFoodSettings(before, MoveUp, nil, FoodSettings{MinimumFood: 1, FoodSpawnChance: 0})
	logNextState(t, "Food minimum enforced", before, MoveUp, after)

	if len(after.Food) < 1 {
		t.Fatalf("food len=%d want>=1", len(after.Food))
	}
	// Ensure food doesn't spawn on the snake.
	occ := map[[2]int]bool{}
	for _, p := range after.Snakes[0].Body {
		occ[[2]int{int(p.X), int(p.Y)}] = true
	}
	for _, f := range after.Food {
		if occ[[2]int{int(f.X), int(f.Y)}] {
			t.Fatalf("food spawned on snake at (%d,%d)", f.X, f.Y)
		}
	}
}

func TestFood_SpawnChanceCanAddExtra(t *testing.T) {
	before := &GameState{
		Width:  5,
		Height: 5,
		YouId:  "me",
		Snakes: []Snake{{Id: "me", Health: 100, Body: []Point{{X: 2, Y: 2}, {X: 2, Y: 2}, {X: 2, Y: 2}}}},
		Food:   []Point{{X: 0, Y: 0}},
		Turn:   0,
	}

	// Force spawn every turn via 100% chance.
	after := NextStateWithFoodSettings(before, MoveUp, nil, FoodSettings{MinimumFood: 0, FoodSpawnChance: 100})
	logNextState(t, "Food spawn chance", before, MoveUp, after)

	if len(after.Food) != 2 {
		t.Fatalf("food len=%d want=2", len(after.Food))
	}
}

// TestGetLegalMovesWithTailDecrement verifies that snakes can move into tail positions
// that will be vacated by the end of the turn.
func TestGetLegalMovesWithTailDecrement(t *testing.T) {
	// Scenario: Snake A's head is next to Snake B's tail.
	// With tail decrement, Snake A should be able to move into that position.
	//
	// Board (5x5):
	//   . . . . .
	//   . . . . .
	//   . . . . .
	//   . H b b .   <- Snake B: body at (2,1), (3,1), (4,1) - tail at (4,1)
	//   . . H . .   <- Snake A: head at (2,0), wants to go Right into (3,0) or Up into (2,1)
	//
	// Without tail decrement: A cannot move Right (blocked by nothing), but cannot move Up into B's body
	// With tail decrement: A can move Up into (2,1) because B's tail at (4,1) will move, so (2,1) becomes the new tail
	// Wait, that doesn't make sense. Let me redo this.
	//
	// Actually the point is: if A's head is adjacent to B's TAIL, A can move there.
	// Let me set up: A's head at (3,1), B's tail at (4,1)
	//
	// Board (5x5):
	//   . . . . .
	//   . . . . .
	//   . . . . .
	//   . . H b T   <- Snake B: head at (2,1), body (3,1), tail at (4,1)
	//   . . . H .   <- Snake A: head at (3,0)
	//
	// A wants to move Up to (3,1) - this is B's body, not allowed even with tail decrement
	// A wants to move Right to (4,0) - this is fine
	//
	// Better scenario: A's head adjacent to B's tail directly
	//
	// Board (5x5):
	//   . . . . .
	//   . . . . .
	//   . . . . .
	//   . H b T .   <- Snake B: head at (1,1), body (2,1), tail at (3,1)
	//   . . . H .   <- Snake A: head at (3,0), wants to move Up to (3,1) which is B's tail
	//
	// Without tail decrement: blocked
	// With tail decrement: allowed (B's tail will move to (2,1), freeing (3,1))

	state := &GameState{
		Width:  5,
		Height: 5,
		YouId:  "a",
		Snakes: []Snake{
			{Id: "a", Health: 100, Body: []Point{{X: 3, Y: 0}, {X: 2, Y: 0}, {X: 1, Y: 0}}}, // A: head at (3,0)
			{Id: "b", Health: 100, Body: []Point{{X: 1, Y: 1}, {X: 2, Y: 1}, {X: 3, Y: 1}}}, // B: head at (1,1), tail at (3,1)
		},
		Food: []Point{},
		Turn: 0,
	}

	t.Logf("Testing tail decrement:\n%s", dumpState(state))

	// Without tail decrement (conservative)
	movesConservative := GetLegalMoves(state)
	t.Logf("GetLegalMoves (conservative): %v", movesConservative)

	// With tail decrement
	movesWithTail := GetLegalMovesWithTailDecrement(state)
	t.Logf("GetLegalMovesWithTailDecrement: %v", movesWithTail)

	// MoveUp (0) should be in movesWithTail but not in movesConservative
	hasUpConservative := false
	for _, m := range movesConservative {
		if m == MoveUp {
			hasUpConservative = true
		}
	}

	hasUpWithTail := false
	for _, m := range movesWithTail {
		if m == MoveUp {
			hasUpWithTail = true
		}
	}

	if hasUpConservative {
		t.Error("Conservative GetLegalMoves should NOT allow MoveUp into B's tail")
	}

	if !hasUpWithTail {
		t.Error("GetLegalMovesWithTailDecrement SHOULD allow MoveUp into B's tail")
	}
}

// TestGetLegalMovesWithTailDecrement_StackedTail verifies that stacked tails (from eating)
// are NOT treated as moving away.
func TestGetLegalMovesWithTailDecrement_StackedTail(t *testing.T) {
	// Snake B just ate, so its last two body segments are the same (stacked tail).
	// A should NOT be able to move into that position.
	//
	// Board (5x5):
	//   . . . . .
	//   . . . . .
	//   . . . . .
	//   . H b T .   <- Snake B: head at (1,1), body (2,1), tail at (3,1) BUT stacked (two at 3,1)
	//   . . . H .   <- Snake A: head at (3,0)

	state := &GameState{
		Width:  5,
		Height: 5,
		YouId:  "a",
		Snakes: []Snake{
			{Id: "a", Health: 100, Body: []Point{{X: 3, Y: 0}, {X: 2, Y: 0}, {X: 1, Y: 0}}},
			{Id: "b", Health: 100, Body: []Point{{X: 1, Y: 1}, {X: 2, Y: 1}, {X: 3, Y: 1}, {X: 3, Y: 1}}}, // Stacked tail!
		},
		Food: []Point{},
		Turn: 0,
	}

	t.Logf("Testing stacked tail:\n%s", dumpState(state))

	movesWithTail := GetLegalMovesWithTailDecrement(state)
	t.Logf("GetLegalMovesWithTailDecrement: %v", movesWithTail)

	// MoveUp should NOT be allowed because B's tail is stacked
	for _, m := range movesWithTail {
		if m == MoveUp {
			t.Error("GetLegalMovesWithTailDecrement should NOT allow MoveUp into B's stacked tail")
		}
	}
}
