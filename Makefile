CONTAINER_NAME = pi-manager

build:
	@echo "Building Docker image..."
	docker-compose build

run:
	@echo "Starting Docker container..."
	docker-compose up -d

shell:
	@echo "Accessing the shell inside the container..."
	docker exec -it $(CONTAINER_NAME) bash

down:
	@echo "Stopping Docker container..."
	docker-compose down