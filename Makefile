.PHONY: all install-deps init generate

export PATH := $(HOME)/go/bin:$(PATH)

VENV_DIR = .venv
PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip

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
	$(PYTHON) trainer/server.py

run-go:
	export LD_LIBRARY_PATH=$(PWD):$$(find $(PWD)/.venv -name "lib" -type d | tr '\n' ':') && \
	go run ./executor

train: $(VENV_DIR)
	$(PYTHON) trainer/train.py

export-onnx: $(VENV_DIR)
	$(PYTHON) trainer/export_onnx.py

generate:
	@mkdir -p data/generated
	export LD_LIBRARY_PATH=$(PWD):$$(find $(PWD)/.venv -name "lib" -type d | tr '\n' ':') && \
	go run ./executor -out-dir data/generated

# Scraper targets
scrape:
	go run ./scraper -out-dir=data -log-path=scraper-data/written_games.log

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
	docker compose up -d --build

docker-scraper-down:
	docker compose down
