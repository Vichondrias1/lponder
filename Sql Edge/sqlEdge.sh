#!/bin/bash

# Create the directory with appropriate permissions
sudo mkdir stirling-pdf

# Change ownership of the directory to the current user
sudo chown $(whoami):$(whoami) stirling-pdf

# Navigate to the directory
cd stirling-pdf

# Pause for 3 seconds
sleep 3

# Download the yml file to the current directory
curl -o docker-compose.yml https://raw.githubusercontent.com/Vichondrias1/lponder/main/Sql-E/docker-compose.yml

# Execute the yml file
docker compose up
