CONTAINER_NAME = pi-manager

build:
	@echo "Building Docker image..."
	docker-compose build

run:
	@echo "Starting Docker container..."
	docker-compose up -d

down:
	@echo "Stopping Docker container..."
	docker-compose down

shell:
	@echo "Accessing the shell inside the container..."
	docker exec -it $(CONTAINER_NAME) bash

restart:
	@echo "Restarting container..."
	make down && make run

test:
	@echo "Running all tests..."
	docker exec -it $(CONTAINER_NAME) pytest tests/

test-api:
	@echo "Running API tests..."
	docker exec -it $(CONTAINER_NAME) pytest tests/api
