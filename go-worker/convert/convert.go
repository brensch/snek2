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

	// Find You and Enemies
	var you *pb.Snake
	var enemies []*pb.Snake
	for _, s := range state.Snakes {
		if s.Id == state.YouId {
			you = s
		} else {
			enemies = append(enemies, s)
		}
	}

	// Sort enemies by ID for consistency across turns
	sort.Slice(enemies, func(i, j int) bool {
		return enemies[i].Id < enemies[j].Id
	})

	if you != nil {
		// Channel 1: My Head
		if len(you.Body) > 0 {
			head := you.Body[0]
			set(1, int(head.X), int(head.Y), 1.0)
		}

		// Channel 2: My Body (excluding head and tail)
		// If length is 1 or 2, this might be empty, which is correct.
		if len(you.Body) > 2 {
			for _, p := range you.Body[1 : len(you.Body)-1] {
				set(2, int(p.X), int(p.Y), 1.0)
			}
		}

		// Channel 3: My Tail
		if len(you.Body) > 1 {
			tail := you.Body[len(you.Body)-1]
			set(3, int(tail.X), int(tail.Y), 1.0)
		}

		// Channel 4: My Health (Plane)
		healthVal := float32(you.Health) / 100.0
		for y := 0; y < Height; y++ {
			for x := 0; x < Width; x++ {
				set(4, x, y, healthVal)
			}
		}
	}

	// Channels 5-16: Enemies (Max 4)
	for i := 0; i < 4 && i < len(enemies); i++ {
		enemy := enemies[i]

		// Head: Ch 5 + i
		if len(enemy.Body) > 0 {
			head := enemy.Body[0]
			set(5+i, int(head.X), int(head.Y), 1.0)
		}

		// Body: Ch 9 + i
		if len(enemy.Body) > 2 {
			for _, p := range enemy.Body[1 : len(enemy.Body)-1] {
				set(9+i, int(p.X), int(p.Y), 1.0)
			}
		}

		// Tail: Ch 13 + i
		if len(enemy.Body) > 1 {
			tail := enemy.Body[len(enemy.Body)-1]
			set(13+i, int(tail.X), int(tail.Y), 1.0)
		}
	}

	return dataPtr
}
