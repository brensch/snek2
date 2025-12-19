.PHONY: all install-deps proto proto-go proto-py init

export PATH := $(HOME)/go/bin:$(PATH)

VENV_DIR = .venv
PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip

all: proto

init:
	go mod init github.com/brensch/snek2 || true
	go mod tidy

$(VENV_DIR):
	python3 -m venv $(VENV_DIR)

install-deps: $(VENV_DIR)
	@echo "Installing system dependencies (requires sudo)..."
	sudo apt-get update && sudo apt-get install -y protobuf-compiler python3-pip python3-venv
	@echo "Installing Go plugins..."
	# Ensure GOPATH/bin is in PATH for the current shell session if needed, 
	# but for installation we just run go install.
	go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
	go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
	@echo "Installing Python dependencies..."
	$(PIP) install grpcio-tools

proto: proto-go proto-py

proto-go:
	@echo "Generating Go code..."
	mkdir -p gen/go
	protoc -Iproto --go_out=gen/go --go_opt=paths=source_relative \
		--go-grpc_out=gen/go --go-grpc_opt=paths=source_relative \
		snake.proto

proto-py: $(VENV_DIR)
	@echo "Generating Python code..."
	mkdir -p gen/python
	$(PYTHON) -m grpc_tools.protoc -Iproto --python_out=gen/python --grpc_python_out=gen/python --pyi_out=gen/python snake.proto

run-py: $(VENV_DIR)
	$(PYTHON) py-inference/server.py

run-go:
	export LD_LIBRARY_PATH=$(PWD):$$(find $(PWD)/.venv -name "lib" -type d | tr '\n' ':') && \
	go run ./go-worker

train: $(VENV_DIR)
	$(PYTHON) py-inference/train.py

run:
	@echo "Starting Snek2 in tmux..."
	tmux new-session -d -s snek '$(MAKE) run-py'
	tmux split-window -h -t snek '$(MAKE) run-go'
	tmux attach -t snek

# Scraper targets
scrape:
	go run ./scraper -db=battlesnake.db -workers=4 -max-players=50

scrape-daemon:
	go run ./scraper -db=battlesnake.db -daemon -interval=30m -auto-export -output-dir=data

scrape-stats:
	go run ./scraper -db=battlesnake.db -stats

scrape-export:
	go run ./scraper -db=battlesnake.db -export -export-path=data/scraped_training.pb -export-max=100

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
		-e MAX_PLAYERS=100 \
		-e INTERVAL=30m \
		-e AUTO_EXPORT=true \
		battlesnake-scraper

docker-scraper-logs:
	docker logs -f battlesnake-scraper

docker-scraper-stop:
	docker stop battlesnake-scraper && docker rm battlesnake-scraper

# Or use docker-compose
docker-scraper-up:
	docker-compose -f docker-compose.scraper.yml up -d --build

docker-scraper-down:
	docker-compose -f docker-compose.scraper.yml down
