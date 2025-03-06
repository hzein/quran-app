#!/bin/bash

# Stop and remove containers, networks, and volumes defined in docker-compose-dev.yml
docker compose -f docker-compose-dev.yml down

# Start containers defined in docker-compose-dev.yml, rebuilding if necessary
docker compose -f docker-compose-dev.yml up --build