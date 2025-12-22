package selfplay

import (
	"fmt"
	"log"
	"strings"

	"github.com/brensch/snek2/game"
)

func PrintBoard(state *game.GameState) {
	// Create a grid
	grid := make([][]string, state.Height)
	for y := range grid {
		grid[y] = make([]string, state.Width)
		for x := range grid[y] {
			grid[y][x] = "."
		}
	}

	// Place Food
	for _, f := range state.Food {
		if isBounds(state, int(f.X), int(f.Y)) {
			grid[f.Y][f.X] = "F"
		}
	}

	// Place Snakes
	for _, s := range state.Snakes {
		char := "s"     // enemy body
		headChar := "S" // enemy head
		if s.Id == state.YouId {
			char = "o"     // my body
			headChar = "O" // my head
		}

		for i, p := range s.Body {
			if !isBounds(state, int(p.X), int(p.Y)) {
				continue
			}
			if i == 0 {
				grid[p.Y][p.X] = headChar
			} else {
				grid[p.Y][p.X] = char
			}
		}
	}

	// Print
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("\n=== TRACE Turn %d (you_id=%s) ===\n", state.Turn, state.YouId))
	for y := state.Height - 1; y >= 0; y-- {
		for x := 0; x < int(state.Width); x++ {
			sb.WriteString(grid[y][x] + " ")
		}
		sb.WriteString("\n")
	}
	log.Print(sb.String())
}

func isBounds(state *game.GameState, x, y int) bool {
	return x >= 0 && x < int(state.Width) && y >= 0 && y < int(state.Height)
}
