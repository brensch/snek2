package store

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// WrittenLog tracks which game IDs have been successfully written.
// It is backed by an append-only log file with one game ID per line.
//
// On startup we read the file into memory for fast dedupe.
// On success we append the game ID and fsync.
//
// The log is intentionally simple and tolerant of partial/corrupt lines.
// If the process crashes mid-write, the next startup may ignore the final partial line.
//
// Format: <game_id>\n
// Note: this is not meant to be a general-purpose WAL; it's just a dedupe list.
type WrittenLog struct {
	mu      sync.RWMutex
	path    string
	file    *os.File
	written map[string]struct{}
}

func OpenWrittenLog(path string) (*WrittenLog, error) {
	written := make(map[string]struct{})

	if path == "" {
		return nil, fmt.Errorf("log path is required")
	}

	// Best-effort load existing IDs.
	if f, err := os.Open(path); err == nil {
		scanner := bufio.NewScanner(f)
		for scanner.Scan() {
			id := strings.TrimSpace(scanner.Text())
			if id == "" {
				continue
			}
			written[id] = struct{}{}
		}
		_ = f.Close()
	}

	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, fmt.Errorf("create log dir: %w", err)
	}

	file, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return nil, fmt.Errorf("open log file: %w", err)
	}

	return &WrittenLog{
		path:    path,
		file:    file,
		written: written,
	}, nil
}

func (l *WrittenLog) Close() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.file == nil {
		return nil
	}
	err := l.file.Close()
	l.file = nil
	return err
}

func (l *WrittenLog) Has(gameID string) bool {
	l.mu.RLock()
	defer l.mu.RUnlock()
	_, ok := l.written[gameID]
	return ok
}

func (l *WrittenLog) Count() int {
	l.mu.RLock()
	defer l.mu.RUnlock()
	return len(l.written)
}

func (l *WrittenLog) SnapshotBoolMap() map[string]bool {
	l.mu.RLock()
	defer l.mu.RUnlock()
	m := make(map[string]bool, len(l.written))
	for id := range l.written {
		m[id] = true
	}
	return m
}

func (l *WrittenLog) Add(gameID string) error {
	if gameID == "" {
		return fmt.Errorf("gameID is empty")
	}

	l.mu.Lock()
	defer l.mu.Unlock()

	if _, ok := l.written[gameID]; ok {
		return nil
	}

	if l.file == nil {
		return fmt.Errorf("log file is closed")
	}

	if _, err := l.file.WriteString(gameID + "\n"); err != nil {
		return fmt.Errorf("append log: %w", err)
	}
	if err := l.file.Sync(); err != nil {
		return fmt.Errorf("sync log: %w", err)
	}

	l.written[gameID] = struct{}{}
	return nil
}

// AddMany appends multiple game IDs and syncs once.
// IDs already present in the log are ignored.
func (l *WrittenLog) AddMany(gameIDs []string) error {
	l.mu.Lock()
	defer l.mu.Unlock()

	if l.file == nil {
		return fmt.Errorf("log file is closed")
	}

	// Filter and write.
	toAdd := 0
	for _, gameID := range gameIDs {
		if gameID == "" {
			continue
		}
		if _, ok := l.written[gameID]; ok {
			continue
		}
		if _, err := l.file.WriteString(gameID + "\n"); err != nil {
			return fmt.Errorf("append log: %w", err)
		}
		l.written[gameID] = struct{}{}
		toAdd++
	}

	if toAdd == 0 {
		return nil
	}
	if err := l.file.Sync(); err != nil {
		return fmt.Errorf("sync log: %w", err)
	}
	return nil
}
