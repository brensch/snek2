package downloader

import (
	"encoding/binary"
	"math"
	"sort"
)

const (
	Width         = 11
	Height        = 11
	Channels      = 10
	BytesPerFloat = 4
	BufferSize    = Channels * Width * Height * BytesPerFloat
)

// EgoStateToBytes builds an ego-centric tensor compatible with the selfplay encoding.
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

	encodeSnakeTTLAndHealth := func(ttlC int, healthC int, s *SnakeData) {
		if s == nil || s.Health <= 0 {
			return
		}
		health := float32(s.Health) / 100.0
		fillPlane(healthC, health)

		if len(s.Body) == 0 {
			return
		}
		l := len(s.Body)
		if l <= 0 {
			return
		}
		denom := float32(l)
		for i, p := range s.Body {
			if p.X < 0 || p.X >= Width || p.Y < 0 || p.Y >= Height {
				continue
			}
			// Head i=0 => 1.0, tail i=l-1 => 1/l.
			ttl := float32(l-i) / denom
			set(ttlC, p.X, p.Y, ttl)
		}
	}

	// Channel layout (10 total):
	// 0 Food
	// 1 Hazards
	// 2 Ego body TTL
	// 3 Enemy 1 body TTL
	// 4 Enemy 2 body TTL
	// 5 Enemy 3 body TTL
	// 6 Ego health plane
	// 7 Enemy 1 health plane
	// 8 Enemy 2 health plane
	// 9 Enemy 3 health plane
	encodeSnakeTTLAndHealth(2, 6, ego)
	for i := 0; i < 3; i++ {
		var s *SnakeData
		if i < len(enemies) {
			s = &enemies[i]
		}
		encodeSnakeTTLAndHealth(3+i, 7+i, s)
	}

	return data
}
