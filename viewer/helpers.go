package main

import (
	"encoding/json"
	"net/http"
	"strconv"
	"strings"
)

func withCORS(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	enc := json.NewEncoder(w)
	_ = enc.Encode(v)
}

func parseIntQuery(r *http.Request, key string, def int) int {
	v := strings.TrimSpace(r.URL.Query().Get(key))
	if v == "" {
		return def
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return def
	}
	if n < 0 {
		return def
	}
	return n
}

func parseInt64Query(r *http.Request, key string, def int64) int64 {
	v := strings.TrimSpace(r.URL.Query().Get(key))
	if v == "" {
		return def
	}
	n, err := strconv.ParseInt(v, 10, 64)
	if err != nil {
		return def
	}
	return n
}

func zipPoints(xs, ys []int32) []Point {
	n := len(xs)
	if len(ys) < n {
		n = len(ys)
	}
	out := make([]Point, 0, n)
	for i := 0; i < n; i++ {
		out = append(out, Point{X: xs[i], Y: ys[i]})
	}
	return out
}

func asInt32Slice(v any) []int32 {
	if v == nil {
		return nil
	}
	switch vv := v.(type) {
	case []int32:
		return vv
	case []int64:
		out := make([]int32, 0, len(vv))
		for _, x := range vv {
			out = append(out, int32(x))
		}
		return out
	case []any:
		out := make([]int32, 0, len(vv))
		for _, x := range vv {
			out = append(out, int32(asInt64(x)))
		}
		return out
	default:
		return nil
	}
}

func asInt64(v any) int64 {
	switch t := v.(type) {
	case int64:
		return t
	case int32:
		return int64(t)
	case int:
		return int64(t)
	case uint64:
		return int64(t)
	case float64:
		return int64(t)
	default:
		return 0
	}
}

func asFloat32(v any) float32 {
	switch t := v.(type) {
	case float32:
		return t
	case float64:
		return float32(t)
	case int64:
		return float32(t)
	case int32:
		return float32(t)
	default:
		return 0
	}
}

func asFloat32Slice(v any) []float32 {
	if v == nil {
		return nil
	}
	switch vv := v.(type) {
	case []float32:
		return vv
	case []float64:
		out := make([]float32, 0, len(vv))
		for _, x := range vv {
			out = append(out, float32(x))
		}
		return out
	case []any:
		out := make([]float32, 0, len(vv))
		for _, x := range vv {
			out = append(out, asFloat32(x))
		}
		return out
	default:
		return nil
	}
}

func asBool(v any) bool {
	switch t := v.(type) {
	case bool:
		return t
	case int64:
		return t != 0
	case int32:
		return t != 0
	default:
		return false
	}
}

func asString(v any) string {
	switch t := v.(type) {
	case string:
		return t
	case []byte:
		return string(t)
	default:
		return ""
	}
}

func asSnakes(v any) []Snake {
	if v == nil {
		return nil
	}
	list, ok := v.([]any)
	if !ok {
		if l2, ok2 := v.([]interface{}); ok2 {
			list = make([]any, len(l2))
			copy(list, l2)
		} else {
			return nil
		}
	}

	snakes := make([]Snake, 0, len(list))
	for _, it := range list {
		m, ok := it.(map[string]any)
		if !ok {
			if m2, ok2 := it.(map[string]interface{}); ok2 {
				m = make(map[string]any, len(m2))
				for k, v := range m2 {
					m[k] = v
				}
			} else {
				continue
			}
		}

		bodyX := asInt32Slice(m["body_x"])
		bodyY := asInt32Slice(m["body_y"])
		s := Snake{
			ID:          asString(m["id"]),
			Alive:       asBool(m["alive"]),
			Health:      int32(asInt64(m["health"])),
			Policy:      int32(asInt64(m["policy"])),
			PolicyProbs: asFloat32Slice(m["policy_probs"]),
			Value:       asFloat32(m["value"]),
			Body:        zipPoints(bodyX, bodyY),
		}
		snakes = append(snakes, s)
	}
	return snakes
}
