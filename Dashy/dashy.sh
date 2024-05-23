#!/bin/bash

# Create the directory with appropriate permissions
sudo mkdir dashy

# Change ownership of the directory to the current user
sudo chown $(whoami):$(whoami) dashy

# Navigate to the directory
cd dashy

# Download the yml file to the current directory
curl -o docker-compose.yml https://raw.githubusercontent.com/Vichondrias1/lponder/main/Dashy/docker-compose.yml

# Create the directory with appropriate permissions
sudo mkdir user-data

# Change ownership of the directory to the current user
sudo chown $(whoami):$(whoami) user-data

# Navigate to the directory
cd user-data

# Download the yml file to the current directory
curl -o conf.yml https://raw.githubusercontent.com/Lissy93/dashy/master/user-data/conf.yml

#back to Dashy Folder
cd ..

# Execute the yml file
docker compose up
