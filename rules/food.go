package rules

import (
	"encoding/binary"
	"hash/fnv"
	"math/rand"

	"github.com/brensch/snek2/game"
)

// FoodSettings matches the common Battlesnake server knobs:
// - MinimumFood: ensure at least this many food items exist after each turn
// - FoodSpawnChance: percentage chance (0-100) to spawn one extra food each turn
//
// Battlesnake engine defaults are commonly MinimumFood=1 and FoodSpawnChance=15.
// (These values are configurable in the official server.)
//
// Note: Our rules package uses a RNG parameter so callers can choose:
// - true randomness for self-play generation, or
// - deterministic pseudo-randomness for MCTS / tests.
//
// This keeps the transition function stable and debuggable.
type FoodSettings struct {
	MinimumFood     int
	FoodSpawnChance int
}

var DefaultFoodSettings = FoodSettings{MinimumFood: 1, FoodSpawnChance: 15}

func applyFoodRules(state *game.GameState, rng *rand.Rand, settings FoodSettings, salt uint64) {
	if state == nil || state.Width <= 0 || state.Height <= 0 {
		return
	}
	if settings.MinimumFood < 0 {
		settings.MinimumFood = 0
	}
	if settings.FoodSpawnChance < 0 {
		settings.FoodSpawnChance = 0
	}
	if settings.FoodSpawnChance > 100 {
		settings.FoodSpawnChance = 100
	}

	// Determine whether we will spawn any food BEFORE doing expensive work.
	deficit := settings.MinimumFood - len(state.Food)
	if deficit < 0 {
		deficit = 0
	}

	spawnExtra := false
	if settings.FoodSpawnChance > 0 {
		if rng != nil {
			spawnExtra = rng.Intn(100) < settings.FoodSpawnChance
		} else {
			// Fast deterministic decision (avoid hashing full bodies every call).
			spawnExtra = int(deterministicU64Fast(state, salt)%100) < settings.FoodSpawnChance
		}
	}

	toSpawn := deficit
	if spawnExtra {
		toSpawn++
	}
	if toSpawn == 0 {
		return
	}

	// We need to actually place food.
	if rng == nil {
		seed := int64(deterministicU64Fast(state, salt))
		if seed == 0 {
			seed = 1
		}
		rng = rand.New(rand.NewSource(seed))
	}

	occupied := make(map[uint64]struct{}, int(state.Width*state.Height))
	key := func(p game.Point) uint64 {
		return (uint64(uint32(p.X)) << 32) | uint64(uint32(p.Y))
	}

	for _, s := range state.Snakes {
		if s.Health <= 0 {
			continue
		}
		for _, p := range s.Body {
			occupied[key(p)] = struct{}{}
		}
	}
	for _, f := range state.Food {
		occupied[key(f)] = struct{}{}
	}

	available := make([]game.Point, 0, int(state.Width*state.Height))
	for y := int32(0); y < state.Height; y++ {
		for x := int32(0); x < state.Width; x++ {
			p := game.Point{X: x, Y: y}
			if _, ok := occupied[key(p)]; ok {
				continue
			}
			available = append(available, p)
		}
	}

	spawnOne := func() {
		if len(available) == 0 {
			return
		}
		i := rng.Intn(len(available))
		state.Food = append(state.Food, available[i])
		// remove chosen slot
		available[i] = available[len(available)-1]
		available = available[:len(available)-1]
	}

	for deficit > 0 {
		spawnOne()
		deficit--
		if len(available) == 0 {
			break
		}
	}

	if spawnExtra {
		spawnOne()
	}
}

// ApplyFoodSettings applies Battlesnake-style food spawning to an existing state.
// This is useful for initialization (e.g. ensure MinimumFood at game start).
func ApplyFoodSettings(state *game.GameState, rng *rand.Rand, settings FoodSettings) {
	applyFoodRules(state, rng, settings, 0x464F4F445F494E49) // "FOOD_INI" salt
}

func deterministicU64Fast(state *game.GameState, salt uint64) uint64 {
	// Intentionally cheap: used on the hot MCTS path.
	// We mix turn + board size + snake head positions + food count.
	h := fnv.New64a()
	var buf [8]byte

	binary.LittleEndian.PutUint64(buf[:], uint64(uint32(state.Width))|(uint64(uint32(state.Height))<<32))
	_, _ = h.Write(buf[:])
	binary.LittleEndian.PutUint64(buf[:], uint64(uint32(state.Turn)))
	_, _ = h.Write(buf[:])
	binary.LittleEndian.PutUint64(buf[:], salt)
	_, _ = h.Write(buf[:])
	binary.LittleEndian.PutUint64(buf[:], uint64(len(state.Food)))
	_, _ = h.Write(buf[:])

	for _, s := range state.Snakes {
		if s.Health <= 0 || len(s.Body) == 0 {
			continue
		}
		_, _ = h.Write([]byte(s.Id))
		head := s.Body[0]
		binary.LittleEndian.PutUint64(buf[:], (uint64(uint32(head.X))<<32)|uint64(uint32(head.Y)))
		_, _ = h.Write(buf[:])
	}

	return h.Sum64()
}
