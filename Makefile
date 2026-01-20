.PHONY: all install-deps init generate

export PATH := $(HOME)/go/bin:$(PATH)

VENV_DIR = .venv
PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip

# Executor tuning (override like: make generate WORKERS=256 ONNX_BATCH_SIZE=512)
OUT_DIR ?= data/generated
WORKERS ?= 512
GAMES_PER_FLUSH ?= 50
ONNX_SESSIONS ?= 1
# Default batch size target.
# Note: effective batch is limited by available in-flight inference requests.
ONNX_BATCH_SIZE ?= 512
ONNX_BATCH_TIMEOUT ?= 5ms
MAX_GAMES ?= 0
MCTS_SIMS ?= 800
TRACE ?= true

EXECUTOR_BIN ?= bin/executor

all:

init:
	go mod init github.com/brensch/snek2 || true
	go mod tidy

$(VENV_DIR):
	python3 -m venv $(VENV_DIR)

install-deps: $(VENV_DIR)
	@echo "Installing system dependencies (requires sudo)..."
	sudo apt-get update && sudo apt-get install -y python3-pip python3-venv wget

install-onnx:
	@echo "Downloading ONNX Runtime GPU library..."
	wget -O onnxruntime-linux-x64-gpu-1.23.2.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-x64-gpu-1.23.2.tgz
	tar -xzf onnxruntime-linux-x64-gpu-1.23.2.tgz
	cp onnxruntime-linux-x64-gpu-1.23.2/lib/libonnxruntime.so.1.23.2 .
	ln -sf libonnxruntime.so.1.23.2 libonnxruntime.so.1
	rm -rf onnxruntime-linux-x64-gpu-1.23.2.tgz onnxruntime-linux-x64-gpu-1.23.2

run-py: $(VENV_DIR)
	PYTHONUNBUFFERED=1 $(PYTHON) -u trainer/server.py

run-go:
	export LD_LIBRARY_PATH=$(PWD):$$(find $(PWD)/.venv -name "lib" -type d | tr '\n' ':') && \
	$(EXECUTOR_BIN) \
		-workers $(WORKERS) \
		-trace=$(TRACE) \
		-mcts-sims $(MCTS_SIMS) \
		-onnx-sessions $(ONNX_SESSIONS) \
		-onnx-batch-size $(ONNX_BATCH_SIZE) \
		-onnx-batch-timeout $(ONNX_BATCH_TIMEOUT)

run-viewer-api:
	export LD_LIBRARY_PATH=$(PWD):$$(find $(PWD)/.venv -name "lib" -type d | tr '\n' ':') && \
	go run ./viewer -listen 127.0.0.1:8080

run-viewer-web:
	cd viewer/web && npm run dev

run-viewer: run-viewer-api

$(EXECUTOR_BIN):
	@mkdir -p $(@D)
	go build -o $(EXECUTOR_BIN) ./executor

train: $(VENV_DIR)
	PYTHONUNBUFFERED=1 $(PYTHON) -u trainer/train.py $(ARGS)

export-onnx: $(VENV_DIR)
	PYTHONUNBUFFERED=1 $(PYTHON) -u trainer/export_onnx.py --in-channels 10 --dtype fp16-f32io --out models/snake_net_fp16_f32io.onnx
	ln -sf snake_net_fp16_f32io.onnx models/snake_net.onnx

# Export ONNX from a specific checkpoint.
# Usage: make export-onnx-ckpt CKPT=models/history/model_x.pt OUT=models/history/model_x.onnx
.PHONY: export-onnx-ckpt
export-onnx-ckpt: $(VENV_DIR)
	@test -n "$(CKPT)" || (echo "CKPT is required" && exit 2)
	@test -n "$(OUT)" || (echo "OUT is required" && exit 2)
	PYTHONUNBUFFERED=1 $(PYTHON) -u trainer/export_onnx.py --in-channels 10 --dtype fp16-f32io --ckpt "$(CKPT)" --out "$(OUT)"

init-ckpt: $(VENV_DIR)
	PYTHONUNBUFFERED=1 $(PYTHON) -u trainer/init_ckpt.py --out models/latest.pt --in-channels 10

reset:
	rm -f models/latest.pt models_bak/latest.pt models/snake_net_fp16_f32io.onnx models/snake_net.onnx
	rm -rf data/generated/tmp data/generated/*.parquet 2>/dev/null || true

bootstrap: reset init-ckpt export-onnx
	$(MAKE) generate WORKERS=32 GAMES_PER_FLUSH=10 MAX_GAMES=200 MCTS_SIMS=64 TRACE=false
	$(MAKE) train ARGS="--epochs 1 --data-dir data/generated --model-path models/latest.pt"

generate: $(EXECUTOR_BIN)
	@mkdir -p $(OUT_DIR)
	@echo "Running executor with: WORKERS=$(WORKERS) ONNX_SESSIONS=$(ONNX_SESSIONS) ONNX_BATCH_SIZE=$(ONNX_BATCH_SIZE) ONNX_BATCH_TIMEOUT=$(ONNX_BATCH_TIMEOUT) MCTS_SIMS=$(MCTS_SIMS) MAX_GAMES=$(MAX_GAMES)"
	export LD_LIBRARY_PATH=$(PWD):$$(find $(PWD)/.venv -name "lib" -type d | tr '\n' ':') && \
	$(EXECUTOR_BIN) \
		-out-dir $(OUT_DIR) \
		-workers $(WORKERS) \
		-games-per-flush $(GAMES_PER_FLUSH) \
		-trace=$(TRACE) \
		-mcts-sims $(MCTS_SIMS) \
		-max-games $(MAX_GAMES) \
		-onnx-sessions $(ONNX_SESSIONS) \
		-onnx-batch-size $(ONNX_BATCH_SIZE) \
		-onnx-batch-timeout $(ONNX_BATCH_TIMEOUT)

# Scraper targets
scrape:
	go run ./scraper -out-dir=data/scraped -log-path=scraper-data/written_games.log

build-scraper:
	go build -o bin/scraper ./scraper

# Docker scraper targets
docker-scraper-build:
	docker build -t battlesnake-scraper -f scraper/Dockerfile .

docker-scraper-run:
	docker run -d --name battlesnake-scraper \
		-v $(PWD)/scraper-data:/data \
		-v $(PWD)/data:/output \
		-e WORKERS=4 \
		-e INTERVAL=30m \
		-e AUTO_EXPORT=true \
		battlesnake-scraper

docker-scraper-logs:
	docker logs -f battlesnake-scraper

docker-scraper-stop:
	docker stop battlesnake-scraper && docker rm battlesnake-scraper

# Or use docker-compose
docker-scraper-up:
	./scripts/docker-bootstrap.sh
	docker compose up -d --build

docker-up:
	./scripts/docker-bootstrap.sh
	docker compose up -d --build

docker-scraper-down:
	docker compose down

# Local, non-Docker orchestrator loop (generate -> train -> export -> repeat).
.PHONY: orchestrate-local
orchestrate-local:
	bash infrastructure/pipeline/orchestrate_local.sh

# Battlesnake API server
BATTLESNAKE_BIN ?= bin/battlesnake
BATTLESNAKE_PORT ?= 8000
BATTLESNAKE_MCTS_SIMS ?= 10000

$(BATTLESNAKE_BIN):
	@mkdir -p $(@D)
	go build -o $(BATTLESNAKE_BIN) ./battlesnake

build-battlesnake: $(BATTLESNAKE_BIN)

run-battlesnake: $(BATTLESNAKE_BIN)
	export LD_LIBRARY_PATH=$(PWD):$$(find $(PWD)/.venv -name "lib" -type d | tr '\n' ':') && \
	$(BATTLESNAKE_BIN) \
		-listen :$(BATTLESNAKE_PORT) \
		-model-path models/snake_net.onnx \
		-sessions 1 \
		-mcts-sims $(BATTLESNAKE_MCTS_SIMS)
# Battlesnake CLI for local testing
# Install CLI: go install github.com/BattlesnakeOfficial/rules/cli/battlesnake@latest
BATTLESNAKE_CLI ?= $(HOME)/go/bin/battlesnake

# Run a test game with our snake vs itself (requires run-battlesnake running on port 8000)
# Usage: make battlesnake-test
# Or with custom snakes: make battlesnake-test SNAKE_URLS="http://localhost:8000 http://localhost:8001"
SNAKE_URLS ?= http://localhost:8000 http://localhost:8000
SNAKE_NAMES ?= snek2-1 snek2-2

.PHONY: install-battlesnake-cli
install-battlesnake-cli:
	go install github.com/BattlesnakeOfficial/rules/cli/battlesnake@latest

.PHONY: battlesnake-test
battlesnake-test:
	@echo "Running Battlesnake game..."
	@echo "Make sure your snake server(s) are running!"
	$(BATTLESNAKE_CLI) play \
		$(foreach url,$(SNAKE_URLS),-u $(url)) \
		$(foreach name,$(SNAKE_NAMES),-n $(name)) \
		--width 11 --height 11 \
		--timeout 500 \
		--viewmap --color

# Run a game and view in browser
.PHONY: battlesnake-test-browser
battlesnake-test-browser:
	@echo "Running Battlesnake game with browser view..."
	@echo "Make sure your snake server(s) are running!"
	$(BATTLESNAKE_CLI) play \
		$(foreach url,$(SNAKE_URLS),-u $(url)) \
		$(foreach name,$(SNAKE_NAMES),-n $(name)) \
		--width 11 --height 11 \
		--timeout 500 \
		--browser

# Run multiple games and count wins
# Usage: make battlesnake-bench GAMES=10
GAMES ?= 10
.PHONY: battlesnake-bench
battlesnake-bench:
	@echo "Running $(GAMES) Battlesnake games..."
	@wins=0; \
	for i in $$(seq 1 $(GAMES)); do \
		result=$$($(BATTLESNAKE_CLI) play \
			$(foreach url,$(SNAKE_URLS),-u $(url)) \
			$(foreach name,$(SNAKE_NAMES),-n $(name)) \
			--width 11 --height 11 \
			--timeout 500 2>&1 | tail -1); \
		echo "Game $$i: $$result"; \
	done