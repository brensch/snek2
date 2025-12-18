package convert

import (
	"encoding/binary"
	"math"
	"sort"

	"sync"

	pb "github.com/brensch/snek2/gen/go"
)

const (
	Width         = 11
	Height        = 11
	Channels      = 17
	BytesPerFloat = 4
	BufferSize    = Channels * Width * Height * BytesPerFloat
)

var bufferPool = sync.Pool{
	New: func() interface{} {
		b := make([]byte, BufferSize)
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

// StateToBytes flattens the GameState into a byte slice for the inference server.
// Output shape: [Channels, Height, Width] (C, H, W)
// Format: Float32 (Little Endian)
// Returns a pointer to the byte slice. Caller must return it to pool using PutBuffer.
func StateToBytes(state *pb.GameState) *[]byte {
	// Get buffer from pool
	dataPtr := GetBuffer()
	data := *dataPtr

	// Zero out the buffer (important since we reuse it)
	for i := range data {
		data[i] = 0
	}

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

	// Sort all snakes by ID to ensure consistent channel assignment
	snakes := make([]*pb.Snake, len(state.Snakes))
	copy(snakes, state.Snakes)
	sort.Slice(snakes, func(i, j int) bool {
		return snakes[i].Id < snakes[j].Id
	})

	// Channels 1-16: Snakes (Max 4)
	// Each snake gets 4 channels: Head, Body, Tail, Health
	for i := 0; i < 4 && i < len(snakes); i++ {
		s := snakes[i]
		baseCh := 1 + (i * 4)

		// Head
		if len(s.Body) > 0 {
			head := s.Body[0]
			set(baseCh, int(head.X), int(head.Y), 1.0)
		}

		// Body (excluding head and tail)
		if len(s.Body) > 2 {
			for _, p := range s.Body[1 : len(s.Body)-1] {
				set(baseCh+1, int(p.X), int(p.Y), 1.0)
			}
		}

		// Tail
		if len(s.Body) > 1 {
			tail := s.Body[len(s.Body)-1]
			set(baseCh+2, int(tail.X), int(tail.Y), 1.0)
		}

		// Health (Plane)
		healthVal := float32(s.Health) / 100.0
		for y := 0; y < Height; y++ {
			for x := 0; x < Width; x++ {
				set(baseCh+3, x, y, healthVal)
			}
		}
	}

	return dataPtr
}
