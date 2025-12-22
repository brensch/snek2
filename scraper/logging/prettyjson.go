package logging

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

// PrettyJSONHandler is a slog.Handler that prints one JSON object per line,
// using json.MarshalIndent for human-readable output.
//
// It is intentionally simple and geared toward CLI/daemon logs.
//
// Note: this handler is not optimized for throughput.
type PrettyJSONHandler struct {
	w         io.Writer
	mu        *sync.Mutex
	level     slog.Leveler
	addSource bool

	attrs  []slog.Attr
	groups []string
}

func NewPrettyJSONHandler(w io.Writer, opts *slog.HandlerOptions) slog.Handler {
	var level slog.Leveler = slog.LevelInfo
	addSource := false
	if opts != nil {
		if opts.Level != nil {
			level = opts.Level
		}
		addSource = opts.AddSource
	}

	return &PrettyJSONHandler{
		w:         w,
		mu:        &sync.Mutex{},
		level:     level,
		addSource: addSource,
	}
}

func (h *PrettyJSONHandler) Enabled(_ context.Context, level slog.Level) bool {
	return level >= h.level.Level()
}

func (h *PrettyJSONHandler) Handle(_ context.Context, r slog.Record) error {
	payload := make(map[string]any, 6)

	when := r.Time
	if when.IsZero() {
		when = time.Now()
	}
	payload["time"] = when.Format(time.RFC3339Nano)
	payload["level"] = r.Level.String()
	payload["msg"] = r.Message

	if h.addSource {
		payload["source"] = sourceFromPC(r.PC)
	}

	attrs := make([]slog.Attr, 0, len(h.attrs)+8)
	attrs = append(attrs, h.attrs...)
	r.Attrs(func(a slog.Attr) bool {
		attrs = append(attrs, a)
		return true
	})

	for _, a := range attrs {
		addAttr(payload, h.groups, a)
	}

	b, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		// As a last resort, avoid dropping logs.
		b = []byte("{\"time\":" + strconv.Quote(payload["time"].(string)) + ",\"level\":" + strconv.Quote(payload["level"].(string)) + ",\"msg\":" + strconv.Quote(r.Message) + "}")
	}

	h.mu.Lock()
	defer h.mu.Unlock()
	_, err = h.w.Write(append(b, '\n'))
	return err
}

func (h *PrettyJSONHandler) WithAttrs(attrs []slog.Attr) slog.Handler {
	clone := *h
	clone.attrs = append(append([]slog.Attr(nil), h.attrs...), attrs...)
	return &clone
}

func (h *PrettyJSONHandler) WithGroup(name string) slog.Handler {
	if name == "" {
		return h
	}
	clone := *h
	clone.groups = append(append([]string(nil), h.groups...), name)
	return &clone
}

func addAttr(root map[string]any, groups []string, attr slog.Attr) {
	attr = resolveAttr(attr)

	dst := root
	for _, g := range groups {
		m, ok := dst[g].(map[string]any)
		if !ok {
			m = map[string]any{}
			dst[g] = m
		}
		dst = m
	}

	addAttrToMap(dst, attr)
}

func addAttrToMap(dst map[string]any, attr slog.Attr) {
	k := attr.Key
	v := attr.Value.Resolve()

	if v.Kind() == slog.KindGroup {
		child := map[string]any{}
		for _, ga := range v.Group() {
			ga = resolveAttr(ga)
			if ga.Key != "" {
				addAttrToMap(child, ga)
			}
		}
		dst[k] = child
		return
	}

	dst[k] = valueToAny(v)
}

func valueToAny(v slog.Value) any {
	v = v.Resolve()
	switch v.Kind() {
	case slog.KindString:
		return v.String()
	case slog.KindInt64:
		return v.Int64()
	case slog.KindUint64:
		return v.Uint64()
	case slog.KindFloat64:
		return v.Float64()
	case slog.KindBool:
		return v.Bool()
	case slog.KindDuration:
		return v.Duration().String()
	case slog.KindTime:
		return v.Time().Format(time.RFC3339Nano)
	case slog.KindAny:
		return v.Any()
	default:
		return v.String()
	}
}

func resolveAttr(a slog.Attr) slog.Attr {
	if a.Key == "" {
		return a
	}
	a.Value = a.Value.Resolve()
	return a
}

func sourceFromPC(pc uintptr) string {
	if pc == 0 {
		return ""
	}
	frames := runtime.CallersFrames([]uintptr{pc})
	f, _ := frames.Next()
	if f.File == "" {
		return ""
	}
	// Keep it compact.
	file := f.File
	if idx := strings.LastIndexByte(file, '/'); idx >= 0 {
		file = file[idx+1:]
	}
	return file + ":" + strconv.Itoa(f.Line)
}
