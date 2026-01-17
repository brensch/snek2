package rules

import (
	"fmt"
	"sort"
	"strings"
	"testing"

	"github.com/brensch/snek2/game"
)

func dumpState(state *game.GameState) string {
	if state == nil {
		return "<nil state>"
	}

	var b strings.Builder
	fmt.Fprintf(&b, "Turn=%d Size=%dx%d You=%s\n", state.Turn, state.Width, state.Height, state.YouId)

	// Food
	fmt.Fprintf(&b, "Food(%d):", len(state.Food))
	for _, f := range state.Food {
		fmt.Fprintf(&b, " (%d,%d)", f.X, f.Y)
	}
	b.WriteString("\n")

	// Snakes (stable order)
	snakes := make([]game.Snake, len(state.Snakes))
	copy(snakes, state.Snakes)
	sort.Slice(snakes, func(i, j int) bool { return snakes[i].Id < snakes[j].Id })
	for _, s := range snakes {
		fmt.Fprintf(&b, "Snake %s Health=%d Len=%d Body:", s.Id, s.Health, len(s.Body))
		for _, p := range s.Body {
			fmt.Fprintf(&b, " (%d,%d)", p.X, p.Y)
		}
		b.WriteString("\n")
	}

	// Simple board view (top-to-bottom)
	w, h := int(state.Width), int(state.Height)
	if w > 0 && h > 0 && w <= 40 && h <= 40 {
		food := make(map[[2]int]bool, len(state.Food))
		for _, f := range state.Food {
			food[[2]int{int(f.X), int(f.Y)}] = true
		}
		occ := make(map[[2]int]int, 64)
		head := make(map[[2]int]bool, 8)
		for _, s := range state.Snakes {
			for i, p := range s.Body {
				k := [2]int{int(p.X), int(p.Y)}
				occ[k]++
				if i == 0 {
					head[k] = true
				}
			}
		}

		b.WriteString("Board:\n")
		for y := h - 1; y >= 0; y-- {
			for x := 0; x < w; x++ {
				k := [2]int{x, y}
				switch {
				case head[k]:
					b.WriteByte('H')
				case food[k] && occ[k] > 0:
					b.WriteByte('*')
				case food[k]:
					b.WriteByte('F')
				case occ[k] > 0:
					c := occ[k]
					if c > 9 {
						c = 9
					}
					b.WriteByte(byte('0' + c))
				default:
					b.WriteByte('.')
				}
			}
			b.WriteByte('\n')
		}
	}

	return b.String()
}

func logNextState(t *testing.T, name string, before *game.GameState, move int, after *game.GameState) {
	t.Helper()
	t.Logf("=== %s ===\nBefore:\n%sMove: %d\nAfter:\n%s", name, dumpState(before), move, dumpState(after))
}

func logNextStateSimultaneous(t *testing.T, name string, before *game.GameState, moves map[string]int, after *game.GameState) {
	t.Helper()
	ids := make([]string, 0, len(moves))
	for id := range moves {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	var mv strings.Builder
	mv.WriteString("Moves:")
	for _, id := range ids {
		fmt.Fprintf(&mv, " %s=%d", id, moves[id])
	}
	mv.WriteByte('\n')
	t.Logf("=== %s ===\nBefore:\n%s%sAfter:\n%s", name, dumpState(before), mv.String(), dumpState(after))
}

func TestNextState_NormalMove_NoFood(t *testing.T) {
	before := &game.GameState{
		Width:  7,
		Height: 7,
		YouId:  "me",
		Snakes: []game.Snake{{
			Id:     "me",
			Health: 10,
			Body:   []game.Point{{X: 3, Y: 3}, {X: 3, Y: 2}, {X: 3, Y: 1}},
		}},
		Food: nil,
		Turn: 0,
	}

	after := NextStateWithFoodSettings(before, MoveUp, nil, FoodSettings{MinimumFood: 0, FoodSpawnChance: 0})
	logNextState(t, "NextState normal move", before, MoveUp, after)

	got := after.Snakes[0].Body
	want := []game.Point{{X: 3, Y: 4}, {X: 3, Y: 3}, {X: 3, Y: 2}}
	if len(got) != len(want) {
		t.Fatalf("body len=%d want=%d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("body[%d]=%v want=%v", i, got[i], want[i])
		}
	}
	if after.Snakes[0].Health != 9 {
		t.Fatalf("health=%d want=9", after.Snakes[0].Health)
	}
}

func TestNextState_EatFood_GrowsByAppendingTail(t *testing.T) {
	// This test encodes the requested rule: do a normal move (tail advances),
	// then grow by duplicating the new tail.
	before := &game.GameState{
		Width:  7,
		Height: 7,
		YouId:  "me",
		Snakes: []game.Snake{{
			Id:     "me",
			Health: 10,
			Body:   []game.Point{{X: 3, Y: 3}, {X: 3, Y: 2}, {X: 3, Y: 1}},
		}},
		Food: []game.Point{{X: 3, Y: 4}},
		Turn: 0,
	}

	after := NextStateWithFoodSettings(before, MoveUp, nil, FoodSettings{MinimumFood: 0, FoodSpawnChance: 0})
	logNextState(t, "NextState eat food", before, MoveUp, after)

	got := after.Snakes[0].Body
	want := []game.Point{{X: 3, Y: 4}, {X: 3, Y: 3}, {X: 3, Y: 2}, {X: 3, Y: 2}}
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
	before := &game.GameState{
		Width:  7,
		Height: 7,
		YouId:  "me",
		Snakes: []game.Snake{{
			Id:     "me",
			Health: 10,
			Body:   []game.Point{{X: 1, Y: 1}, {X: 1, Y: 1}, {X: 1, Y: 1}},
		}},
		Food: []game.Point{{X: 1, Y: 2}},
		Turn: 0,
	}

	after := NextStateWithFoodSettings(before, MoveUp, nil, FoodSettings{MinimumFood: 0, FoodSpawnChance: 0})
	logNextState(t, "NextState stacked spawn eat", before, MoveUp, after)

	got := after.Snakes[0].Body
	want := []game.Point{{X: 1, Y: 2}, {X: 1, Y: 1}, {X: 1, Y: 1}, {X: 1, Y: 1}}
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
	before := &game.GameState{
		Width:  7,
		Height: 7,
		YouId:  "a",
		Snakes: []game.Snake{
			{Id: "a", Health: 10, Body: []game.Point{{X: 1, Y: 1}, {X: 1, Y: 1}, {X: 1, Y: 1}}},
			{Id: "b", Health: 10, Body: []game.Point{{X: 5, Y: 5}, {X: 5, Y: 5}, {X: 5, Y: 5}}},
		},
		Food: []game.Point{{X: 1, Y: 2}},
		Turn: 0,
	}

	moves := map[string]int{"a": MoveUp, "b": MoveLeft}
	after := NextStateSimultaneousWithFoodSettings(before, moves, nil, FoodSettings{MinimumFood: 0, FoodSpawnChance: 0})
	logNextStateSimultaneous(t, "NextStateSimultaneous one eats", before, moves, after)

	// Snake a ate: should grow by duplicating new tail.
	var a, b *game.Snake
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
	wantA := []game.Point{{X: 1, Y: 2}, {X: 1, Y: 1}, {X: 1, Y: 1}, {X: 1, Y: 1}}
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
	wantB := []game.Point{{X: 4, Y: 5}, {X: 5, Y: 5}, {X: 5, Y: 5}}
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
	before := &game.GameState{
		Width:  5,
		Height: 5,
		YouId:  "me",
		Snakes: []game.Snake{{Id: "me", Health: 100, Body: []game.Point{{X: 2, Y: 2}, {X: 2, Y: 2}, {X: 2, Y: 2}}}},
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
	before := &game.GameState{
		Width:  5,
		Height: 5,
		YouId:  "me",
		Snakes: []game.Snake{{Id: "me", Health: 100, Body: []game.Point{{X: 2, Y: 2}, {X: 2, Y: 2}, {X: 2, Y: 2}}}},
		Food:   []game.Point{{X: 0, Y: 0}},
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

	state := &game.GameState{
		Width:  5,
		Height: 5,
		YouId:  "a",
		Snakes: []game.Snake{
			{Id: "a", Health: 100, Body: []game.Point{{X: 3, Y: 0}, {X: 2, Y: 0}, {X: 1, Y: 0}}}, // A: head at (3,0)
			{Id: "b", Health: 100, Body: []game.Point{{X: 1, Y: 1}, {X: 2, Y: 1}, {X: 3, Y: 1}}}, // B: head at (1,1), tail at (3,1)
		},
		Food: []game.Point{},
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

	state := &game.GameState{
		Width:  5,
		Height: 5,
		YouId:  "a",
		Snakes: []game.Snake{
			{Id: "a", Health: 100, Body: []game.Point{{X: 3, Y: 0}, {X: 2, Y: 0}, {X: 1, Y: 0}}},
			{Id: "b", Health: 100, Body: []game.Point{{X: 1, Y: 1}, {X: 2, Y: 1}, {X: 3, Y: 1}, {X: 3, Y: 1}}}, // Stacked tail!
		},
		Food: []game.Point{},
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
