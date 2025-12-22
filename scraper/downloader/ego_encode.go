package downloader

import (
	"encoding/binary"
	"math"
	"sort"
)

const (
	Width         = 11
	Height        = 11
	Channels      = 14
	BytesPerFloat = 4
	BufferSize    = Channels * Width * Height * BytesPerFloat
)

// EgoStateToBytes builds the 14-channel ego-centric tensor described in trainer/model.py.
// Output layout: (C,H,W) float32 little-endian.
func EgoStateToBytes(frame *FrameData, egoID string) []byte {
	data := make([]byte, BufferSize)

	set := func(c, x, y int, val float32) {
		if x < 0 || x >= Width || y < 0 || y >= Height {
			return
		}
		idx := (c*Height*Width + y*Width + x) * BytesPerFloat
		binary.LittleEndian.PutUint32(data[idx:], math.Float32bits(val))
	}

	fillPlane := func(c int, val float32) {
		bits := math.Float32bits(val)
		start := c * Height * Width * BytesPerFloat
		end := start + (Height * Width * BytesPerFloat)
		for i := start; i < end; i += 4 {
			binary.LittleEndian.PutUint32(data[i:], bits)
		}
	}

	// Food
	for _, f := range frame.Food {
		set(0, f.X, f.Y, 1.0)
	}

	// Hazards (some feeds put hazards at frame.Hazards, some under frame.Board.Hazards)
	for _, h := range frame.Hazards {
		set(1, h.X, h.Y, 1.0)
	}
	for _, h := range frame.Board.Hazards {
		set(1, h.X, h.Y, 1.0)
	}

	// Find ego snake and alive enemies
	var ego *SnakeData
	var enemies []SnakeData
	for i := range frame.Snakes {
		s := &frame.Snakes[i]
		if s.Death != nil || s.Health <= 0 || len(s.Body) == 0 {
			continue
		}
		if s.ID == egoID {
			ego = s
		} else {
			enemies = append(enemies, *s)
		}
	}

	// Stable enemy ordering: by ID, take first 3
	sort.Slice(enemies, func(i, j int) bool { return enemies[i].ID < enemies[j].ID })
	if len(enemies) > 3 {
		enemies = enemies[:3]
	}

	// Ego channels
	if ego != nil {
		// Head
		set(2, ego.Body[0].X, ego.Body[0].Y, 1.0)

		// Body counts (including head and tail; stacked segments show as >1)
		counts := make(map[[2]int]int, len(ego.Body))
		for _, p := range ego.Body {
			key := [2]int{p.X, p.Y}
			counts[key]++
		}
		for k, n := range counts {
			set(3, k[0], k[1], float32(n))
		}

		fillPlane(4, float32(ego.Health)/100.0)
	}

	// Enemy channels
	for i, e := range enemies {
		base := 5 + i*3
		set(base, e.Body[0].X, e.Body[0].Y, 1.0) // head
		counts := make(map[[2]int]int, len(e.Body))
		for _, p := range e.Body {
			key := [2]int{p.X, p.Y}
			counts[key]++
		}
		for k, n := range counts {
			set(base+1, k[0], k[1], float32(n))
		}
		fillPlane(base+2, float32(e.Health)/100.0)
	}

	return data
}
