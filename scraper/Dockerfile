# Build stage
FROM golang:1.25.5-bookworm AS builder

WORKDIR /app

# Install SQLite dev libraries for CGO
RUN apt-get update && apt-get install -y gcc libsqlite3-dev

# Copy go mod files first for caching
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build the scraper with CGO enabled (required for SQLite)
RUN CGO_ENABLED=1 GOOS=linux go build -o /scraper ./scraper

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libsqlite3-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy binary from builder
COPY --from=builder /scraper /app/scraper

# Create data directories
RUN mkdir -p /data /output

# Environment variables with defaults
ENV DB_PATH=/data/battlesnake.db
ENV OUTPUT_DIR=/output
ENV WORKERS=4
ENV MAX_PLAYERS=100
ENV DELAY=500ms
ENV INTERVAL=30m
ENV AUTO_EXPORT=true

# Volume mount points
VOLUME ["/data", "/output"]

# Run the scraper
ENTRYPOINT ["/app/scraper"]
CMD ["-daemon", "-auto-export"]
