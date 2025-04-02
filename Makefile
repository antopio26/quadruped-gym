BUILD_TYPE = Release
IMAGE_TYPE = base
TEST_TIME_SECS = 10

CONTAINER_IMAGE := ait4automation
DOCKERFILE := Dockerfile.nvidia
PERCENT := %
ROOT_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

default: build


build: ## Build release container
	@docker build \
		--tag $(CONTAINER_IMAGE) \
		--file $(DOCKERFILE) \
		.

run: ## Run a disposable development container
	@xhost +local:docker
	@docker run -it --user=root \
		--runtime nvidia \
		--rm \
		--network host \
		--ipc=host \
		--gpus all \
		-v ./:/workspace \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-v ~/.Xauthority:/root/.Xauthority \
		-e DISPLAY=$$DISPLAY \
		-e XAUTHORITY=/root/.Xauthority \
		-e __GLX_VENDOR_LIBRARY_NAME=nvidia \
		$(CONTAINER_IMAGE)

clean: ## Clean image artifacts
	-docker rmi $(CONTAINER_IMAGE)


help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "%-10s %s\n", $$1, $$2}'

# Regola "catch-all" per evitare errori sugli obiettivi extra
%:
	@:
.PHONY: default build run clean help
