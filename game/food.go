// food.go implements food spawning mechanics for Battlesnake.

package game

import (
	"math/rand"
)

// FoodSettings controls food spawning behavior.
type FoodSettings struct {
	MinimumFood     int // Guaranteed minimum on board at all times
	FoodSpawnChance int // Percentage chance (0–100) to spawn extra food each turn
}

// DefaultFoodSettings matches standard Battlesnake rules (1 minimum, 15% chance each turn).
var DefaultFoodSettings = FoodSettings{MinimumFood: 1, FoodSpawnChance: 15}

// applyFoodRules spawns food according to settings after a state transition.
// If rng is nil, we use deterministic pseudo-random logic.
func applyFoodRules(state *GameState, rng *rand.Rand, settings FoodSettings, salt uint64) {
	// Build occupancy set (snakes only; existing food already stored in state.Food)
	occupied := make(map[[2]int32]bool)
	for _, s := range state.Snakes {
		for _, p := range s.Body {
			occupied[[2]int32{p.X, p.Y}] = true
		}
	}
	for _, f := range state.Food {
		occupied[[2]int32{f.X, f.Y}] = true
	}

	// Helper to spawn one piece of food, returns false if no room
	spawn := func() bool {
		freeSpots := make([]Point, 0, int(state.Width*state.Height)-len(occupied))
		for y := int32(0); y < state.Height; y++ {
			for x := int32(0); x < state.Width; x++ {
				if !occupied[[2]int32{x, y}] {
					freeSpots = append(freeSpots, Point{X: x, Y: y})
				}
			}
		}
		if len(freeSpots) == 0 {
			return false
		}
		var idx int
		if rng != nil {
			idx = rng.Intn(len(freeSpots))
		} else {
			// Deterministic fallback: hash of turn+salt
			idx = int(deterministicU64Fast(uint64(state.Turn), salt) % uint64(len(freeSpots)))
		}
		p := freeSpots[idx]
		state.Food = append(state.Food, p)
		occupied[[2]int32{p.X, p.Y}] = true
		return true
	}

	// 1. Ensure minimum food
	for len(state.Food) < settings.MinimumFood {
		if !spawn() {
			break
		}
	}

	// 2. Random extra food
	if settings.FoodSpawnChance > 0 {
		var roll int
		if rng != nil {
			roll = rng.Intn(100)
		} else {
			roll = int(deterministicU64Fast(uint64(state.Turn), salt^0xF00D) % 100)
		}
		if roll < settings.FoodSpawnChance {
			spawn()
		}
	}
}

// ApplyFoodSettings is the exported version for testing convenience.
// It invokes applyFoodRules with a default salt.
func ApplyFoodSettings(state *GameState, rng *rand.Rand, settings FoodSettings) {
	applyFoodRules(state, rng, settings, 0xDEADBEEF)
}

// deterministicU64Fast is a simple deterministic hasher for reproducibility.
func deterministicU64Fast(a, b uint64) uint64 {
	// Variant of splitmix64
	x := a + b
	x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9
	x = (x ^ (x >> 27)) * 0x94d049bb133111eb
	return x ^ (x >> 31)
}
