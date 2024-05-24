#!/bin/bash

#login to tailscale using authkey
sudo tailscale up --authkey $tailscaleAuthkey

# Create the directory with appropriate permissions
sudo mkdir -p tailscale

# Change ownership of the directory to the current user
sudo chown $(whoami):$(whoami) tailscale

# Navigate to the directory
cd tailscale

# Download the yml file to the current directory
curl -o docker-compose.yml https://raw.githubusercontent.com/Vichondrias1/lponder/main/Tailscale/docker-compose.yml

# Execute the yml file
docker compose up
