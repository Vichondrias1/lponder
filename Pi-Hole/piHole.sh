#!/bin/bash

# Create the directory with appropriate permissions
sudo mkdir pi-hole

# Change ownership of the directory to the current user
sudo chown $(whoami):$(whoami) pi-hole

# Navigate to the directory
cd pi-hole

# Pause for 3 seconds
sleep 3

# Download the yml file to the current directory
curl -o docker-compose.yml https://raw.githubusercontent.com/Vichondrias1/lponder/main/Pi-Hole/docker-compose.yml

# Execute the yml file
docker compose up
