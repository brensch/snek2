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
	Channels      = 10
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

	// Channel layout (10 total):
	// 0: Food
	// 1: Dangers / hazards (currently unavailable in GameState; left as zeros)
	// 2..5: Snake body TTL planes (ego + up to 3 enemies)
	// 6..9: Snake health planes (ego + up to 3 enemies) where the entire plane is health

	// Channel 0: Food
	for _, p := range state.Food {
		set(0, int(p.X), int(p.Y), 1.0)
	}

	encodeSnakeTTLAndHealth := func(ttlC int, healthC int, s *game.Snake) {
		if s == nil || s.Health <= 0 {
			return
		}
		health := float32(s.Health) / 100.0

		// Health plane: entire channel is health.
		start := healthC * Height * Width
		end := start + (Height * Width)
		for i := start; i < end; i++ {
			data[i] = health
		}

		if len(s.Body) == 0 {
			return
		}

		// TTL plane: per occupied cell store a monotonically decreasing "time-to-live" value
		// from head -> tail. Normalized to (0,1], tail is smallest.
		l := len(s.Body)
		if l <= 0 {
			return
		}
		denom := float32(l)
		for i, p := range s.Body {
			x := int(p.X)
			y := int(p.Y)
			if x < 0 || x >= Width || y < 0 || y >= Height {
				continue
			}
			// Head i=0 => 1.0, tail i=l-1 => 1/l.
			ttl := float32(l-i) / denom
			data[ttlC*Height*Width+y*Width+x] = ttl
		}
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

	// Snake channels: 4 snakes (ego + 3 enemies)
	// TTL:    2..5
	// Health:  6..9
	encodeSnakeTTLAndHealth(2, 6, ego)
	for i := 0; i < 3; i++ {
		var s *game.Snake
		if i < len(enemies) {
			s = enemies[i]
		}
		encodeSnakeTTLAndHealth(3+i, 7+i, s)
	}

	return dataPtr
}

// StateToBytes flattens the GameState into a byte slice for the inference server.
// Output shape: [Channels, Height, Width] (C, H, W)
// Format: Float32 (Little Endian)
// Encoding is ego-centric and uses state.you_id as the ego snake.
// Channels match the Go selfplay/training encoding:
// 0 Food
// 1 Dangers / hazards (unused by executor state today)
// 2 Ego body TTL
// 3 Enemy 1 body TTL
// 4 Enemy 2 body TTL
// 5 Enemy 3 body TTL
// 6 Ego health plane
// 7 Enemy 1 health plane
// 8 Enemy 2 health plane
// 9 Enemy 3 health plane
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

	// Channel 0: Food
	for _, p := range state.Food {
		set(0, int(p.X), int(p.Y), 1.0)
	}

	encodeSnakeTTLAndHealth := func(ttlC int, healthC int, s *game.Snake) {
		if s == nil || s.Health <= 0 {
			return
		}
		health := float32(s.Health) / 100.0

		// Health plane: entire channel is health.
		healthBits := math.Float32bits(health)
		startIdx := healthC * Height * Width * BytesPerFloat
		endIdx := startIdx + (Height * Width * BytesPerFloat)
		for i := startIdx; i < endIdx; i += 4 {
			binary.LittleEndian.PutUint32(data[i:], healthBits)
		}

		if len(s.Body) == 0 {
			return
		}
		l := len(s.Body)
		if l <= 0 {
			return
		}
		denom := float32(l)
		for i, p := range s.Body {
			x := int(p.X)
			y := int(p.Y)
			if x < 0 || x >= Width || y < 0 || y >= Height {
				continue
			}
			ttl := float32(l-i) / denom
			set(ttlC, x, y, ttl)
		}
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

	encodeSnakeTTLAndHealth(2, 6, ego)
	for i := 0; i < 3; i++ {
		var s *game.Snake
		if i < len(enemies) {
			s = enemies[i]
		}
		encodeSnakeTTLAndHealth(3+i, 7+i, s)
	}

	return dataPtr
}
