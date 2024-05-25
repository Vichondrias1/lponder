#!/bin/bash

# Create the directory with appropriate permissions
sudo mkdir nginx-proxy-manager

# Change ownership of the directory to the current user
sudo chown $(whoami):$(whoami) nginx-proxy-manager

# Navigate to the directory
cd nginx-proxy-manager

# Pause for 3 seconds
sleep 3

# Download the yml file to the current directory
curl -o docker-compose.yml https://raw.githubusercontent.com/Vichondrias1/lponder/main/Nginx-Proxy-Manager/docker-compose.yml

# Execute the yml file
docker compose up
