package downloader

import (
	"encoding/binary"
	"math"
	"sort"
)

const (
	Width         = 11
	Height        = 11
	Channels      = 6
	BytesPerFloat = 4
	BufferSize    = Channels * Width * Height * BytesPerFloat
)

// EgoStateToBytes builds the ego-centric tensor described in trainer/model.py.
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

	encodeSnake := func(c int, s *SnakeData) {
		if s == nil || s.Health <= 0 || len(s.Body) == 0 {
			return
		}
		health := float32(s.Health) / 100.0
		fillPlane(c, health)

		counts := make([]float32, Width*Height)
		headMax := make([]float32, Width*Height)
		tailFlag := make([]float32, Width*Height)

		l := len(s.Body)
		denom := float32(1)
		if l > 1 {
			denom = float32(l - 1)
		}
		for i, p := range s.Body {
			if p.X < 0 || p.X >= Width || p.Y < 0 || p.Y >= Height {
				continue
			}
			idx := p.Y*Width + p.X
			counts[idx] += 1.0
			headness := float32(1.0) - (float32(i) / denom) // head=1 .. tail=0
			if headness > headMax[idx] {
				headMax[idx] = headness
			}
			if i == l-1 {
				tailFlag[idx] = 1.0
			}
		}

		// Final per-cell value:
		// - health is present everywhere on the plane (background)
		// - +count encodes stacked segments (start-of-game stack becomes 3)
		// - +headMax helps localize the head
		// - +0.25*tailFlag helps localize tail/persistence (tailFlag=1 and count>1 => persists)
		for y := 0; y < Height; y++ {
			for x := 0; x < Width; x++ {
				idx := y*Width + x
				cCount := counts[idx]
				if cCount == 0 {
					continue
				}
				val := health + cCount + headMax[idx] + 0.25*tailFlag[idx]
				set(c, x, y, val)
			}
		}
	}

	// Snake channels: 1 plane per snake (health * ordered body mask)
	encodeSnake(2, ego)
	for i := range enemies {
		e := &enemies[i]
		encodeSnake(3+i, e)
	}

	return data
}
