// visualize.go - Console visualization for debugging self-play games.
//
// PrintBoard outputs an ASCII representation of the game board and
// neural network input tensor values for debugging and development.
package selfplay

import (
	"fmt"
	"log"
	"strings"

	"github.com/brensch/snek2/executor/convert"
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

	// Also print the model input layers (channels) for this (you_id) perspective.
	printEncodedLayers(&sb, state)
	log.Print(sb.String())
}

func printEncodedLayers(sb *strings.Builder, state *game.GameState) {
	dataPtr := convert.StateToFloat32(state)
	data := *dataPtr
	defer convert.PutFloatBuffer(dataPtr)

	// Channel layout (10 total):
	// 0: Food
	// 1: Dangers / hazards (unused today)
	// 2..5: Snake body TTL planes (ego + up to 3 enemies)
	// 6..9: Snake health planes (ego + up to 3 enemies)
	channelName := func(c int) string {
		switch c {
		case 0:
			return "food"
		case 1:
			return "hazards"
		case 2:
			return "ego_ttl"
		case 3:
			return "enemy1_ttl"
		case 4:
			return "enemy2_ttl"
		case 5:
			return "enemy3_ttl"
		case 6:
			return "ego_health"
		case 7:
			return "enemy1_health"
		case 8:
			return "enemy2_health"
		case 9:
			return "enemy3_health"
		default:
			return "unknown"
		}
	}

	sb.WriteString("\n--- TRACE Encoded input layers (C,H,W) ---\n")
	for c := 0; c < convert.Channels; c++ {
		sb.WriteString(fmt.Sprintf("Layer %d (%s):\n", c, channelName(c)))
		base := c * convert.Height * convert.Width
		for y := convert.Height - 1; y >= 0; y-- {
			for x := 0; x < convert.Width; x++ {
				v := data[base+y*convert.Width+x]
				if v == 0 {
					sb.WriteString("   . ")
					continue
				}
				// Keep it compact but numeric.
				sb.WriteString(fmt.Sprintf("%4.2f ", v))
			}
			sb.WriteString("\n")
		}
	}
}

func isBounds(state *game.GameState, x, y int) bool {
	return x >= 0 && x < int(state.Width) && y >= 0 && y < int(state.Height)
}
