#!/bin/bash

# Create the directory with appropriate permissions
sudo mkdir tailscale

# Change ownership of the directory to the current user
sudo chown $(whoami):$(whoami) tailscale

# Navigate to the directory
cd tailscale

echo "Please provide the tailscale authkey:"

read authkey

export tailscaleAuthkey=$authkey

# Download the yml file to the current directory
curl -o docker-compose.yml https://raw.githubusercontent.com/Vichondrias1/lponder/main/Tailscale/docker-compose.yml


# Execute the yml file
docker compose up
