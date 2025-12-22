# Build stage
FROM golang:1.25.5-bookworm AS builder

WORKDIR /app

# Copy go mod files first for caching
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build the scraper
RUN CGO_ENABLED=0 GOOS=linux go build -o /scraper ./scraper

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy binary from builder
COPY --from=builder /scraper /app/scraper

# Create data directories
RUN mkdir -p /data /output

# Environment variables with defaults
ENV WRITTEN_LOG=/data/written_games.log
ENV OUT_DIR=/output
ENV MAX_PLAYERS=100
ENV DELAY=500ms

# Volume mount points
VOLUME ["/data", "/output"]

# Run the scraper once
ENTRYPOINT ["/app/scraper"]
