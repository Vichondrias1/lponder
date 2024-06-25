#!/bin/bash

# Create the directory with appropriate permissions
sudo mkdir sql-edge

# Change ownership of the directory to the current user
sudo chown $(whoami):$(whoami) sql-edge

# Navigate to the directory
cd sql-edge

# Pause for 3 seconds
sleep 3

# Download the yml file to the current directory
curl -o docker-compose.yml https://raw.githubusercontent.com/Vichondrias1/lponder/main/Sql-Edge/docker-compose.yml

# Execute the yml file
docker compose up
