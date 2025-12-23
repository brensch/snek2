package convert

import (
	"encoding/binary"
	"math"
	"sort"

	"sync"

	"github.com/brensch/snek2/game"
)

const (
	Width         = 11
	Height        = 11
	Channels      = 14
	BytesPerFloat = 4
	BufferSize    = Channels * Width * Height * BytesPerFloat
	FloatSize     = Channels * Width * Height
)

var bufferPool = sync.Pool{
	New: func() interface{} {
		b := make([]byte, BufferSize)
		return &b
	},
}

var floatPool = sync.Pool{
	New: func() interface{} {
		b := make([]float32, FloatSize)
		return &b
	},
}

// GetBuffer returns a buffer from the pool.
func GetBuffer() *[]byte {
	return bufferPool.Get().(*[]byte)
}

// PutBuffer returns a buffer to the pool.
func PutBuffer(b *[]byte) {
	bufferPool.Put(b)
}

func GetFloatBuffer() *[]float32 {
	return floatPool.Get().(*[]float32)
}

func PutFloatBuffer(b *[]float32) {
	floatPool.Put(b)
}

// StateToFloat32 encodes the GameState into a pooled float32 slice suitable for ONNX input.
// Output shape: [Channels, Height, Width] (C, H, W)
// Returns a pointer to the float slice. Caller must return it to pool using PutFloatBuffer.
func StateToFloat32(state *game.GameState) *[]float32 {
	dataPtr := GetFloatBuffer()
	data := *dataPtr
	clear(data)

	set := func(c, x, y int, val float32) {
		if x < 0 || x >= Width || y < 0 || y >= Height {
			return
		}
		idx := (c*Height*Width + y*Width + x)
		data[idx] = val
	}

	fillPlane := func(c int, val float32) {
		start := c * Height * Width
		end := start + (Height * Width)
		for i := start; i < end; i++ {
			data[i] = val
		}
	}

	// Channel 0: Food
	for _, p := range state.Food {
		set(0, int(p.X), int(p.Y), 1.0)
	}

	// Find ego snake and alive enemies
	var ego *game.Snake
	var enemies []*game.Snake
	for i := range state.Snakes {
		s := &state.Snakes[i]
		if s.Health <= 0 || len(s.Body) == 0 {
			continue
		}
		if s.Id == state.YouId {
			ego = s
		} else {
			enemies = append(enemies, s)
		}
	}

	// Stable enemy ordering: by ID, take first 3
	sort.Slice(enemies, func(i, j int) bool { return enemies[i].Id < enemies[j].Id })
	if len(enemies) > 3 {
		enemies = enemies[:3]
	}

	if ego != nil {
		head := ego.Body[0]
		set(2, int(head.X), int(head.Y), 1.0)
		counts := make(map[[2]int]int, len(ego.Body))
		for _, p := range ego.Body {
			counts[[2]int{int(p.X), int(p.Y)}]++
		}
		for k, n := range counts {
			set(3, k[0], k[1], float32(n))
		}
		fillPlane(4, float32(ego.Health)/100.0)
	}

	for i, e := range enemies {
		base := 5 + i*3
		head := e.Body[0]
		set(base, int(head.X), int(head.Y), 1.0)
		counts := make(map[[2]int]int, len(e.Body))
		for _, p := range e.Body {
			counts[[2]int{int(p.X), int(p.Y)}]++
		}
		for k, n := range counts {
			set(base+1, k[0], k[1], float32(n))
		}
		fillPlane(base+2, float32(e.Health)/100.0)
	}

	return dataPtr
}

// StateToBytes flattens the GameState into a byte slice for the inference server.
// Output shape: [Channels, Height, Width] (C, H, W)
// Format: Float32 (Little Endian)
// Encoding is ego-centric and uses state.you_id as the ego snake.
// Channels match trainer/model.py:
// 0 Food
// 1 Hazards (unused by executor state today)
// 2 My Head
// 3 My Body (stacked counts)
// 4 My Health (global plane)
// 5-7 Enemy 1 (head, body, health)
// 8-10 Enemy 2 (head, body, health)
// 11-13 Enemy 3 (head, body, health)
// Returns a pointer to the byte slice. Caller must return it to pool using PutBuffer.
func StateToBytes(state *game.GameState) *[]byte {
	// Get buffer from pool
	dataPtr := GetBuffer()
	data := *dataPtr

	// Zero out the buffer (important since we reuse it)
	clear(data)

	// Helper to set float value at (c, y, x)
	// We use (y, x) to match standard image conventions (Height, Width)
	set := func(c, x, y int, val float32) {
		if x < 0 || x >= Width || y < 0 || y >= Height {
			return
		}
		// Index: (c * H * W + y * W + x) * 4
		idx := (c*Height*Width + y*Width + x) * BytesPerFloat
		binary.LittleEndian.PutUint32(data[idx:], math.Float32bits(val))
	}

	fillPlane := func(c int, val float32) {
		bits := math.Float32bits(val)
		startIdx := c * Height * Width * BytesPerFloat
		endIdx := startIdx + (Height * Width * BytesPerFloat)
		for i := startIdx; i < endIdx; i += 4 {
			binary.LittleEndian.PutUint32(data[i:], bits)
		}
	}

	// Channel 0: Food
	for _, p := range state.Food {
		set(0, int(p.X), int(p.Y), 1.0)
	}

	// Find ego snake and alive enemies
	var ego *game.Snake
	var enemies []*game.Snake
	for i := range state.Snakes {
		s := &state.Snakes[i]
		if s.Health <= 0 || len(s.Body) == 0 {
			continue
		}
		if s.Id == state.YouId {
			ego = s
		} else {
			enemies = append(enemies, s)
		}
	}

	// Stable enemy ordering: by ID, take first 3
	sort.Slice(enemies, func(i, j int) bool { return enemies[i].Id < enemies[j].Id })
	if len(enemies) > 3 {
		enemies = enemies[:3]
	}

	// Ego channels
	if ego != nil {
		// Head
		head := ego.Body[0]
		set(2, int(head.X), int(head.Y), 1.0)

		// Body counts (including head and tail)
		counts := make(map[[2]int]int, len(ego.Body))
		for _, p := range ego.Body {
			counts[[2]int{int(p.X), int(p.Y)}]++
		}
		for k, n := range counts {
			set(3, k[0], k[1], float32(n))
		}

		fillPlane(4, float32(ego.Health)/100.0)
	}

	// Enemy channels
	for i, e := range enemies {
		base := 5 + i*3
		head := e.Body[0]
		set(base, int(head.X), int(head.Y), 1.0)
		counts := make(map[[2]int]int, len(e.Body))
		for _, p := range e.Body {
			counts[[2]int{int(p.X), int(p.Y)}]++
		}
		for k, n := range counts {
			set(base+1, k[0], k[1], float32(n))
		}
		fillPlane(base+2, float32(e.Health)/100.0)
	}

	return dataPtr
}
