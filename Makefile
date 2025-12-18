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
	go run ./go-worker

train: $(VENV_DIR)
	$(PYTHON) py-inference/train.py

run:
	@echo "Starting Snek2 in tmux..."
	tmux new-session -d -s snek '$(MAKE) run-py'
	tmux split-window -h -t snek '$(MAKE) run-go'
	tmux attach -t snek
