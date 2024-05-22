#!/bin/bash

#tried but doesnt work wtff haha!!!
#sudo mkdir uptime-kuma && cd uptime-kuma

sudo mkdir uptime-kuma

sleep 3
#download the yml file to the current directory
curl -o uptime-kuma/docker-compose.yml https://raw.githubusercontent.com/Vichondrias1/lponder/main/docker-compose.yml
#curl -O https://raw.githubusercontent.com/Vichondrias1/lponder/main/docker-compose.yml

#execute the yml file
docker-compose -f uptime-kuma/docker-compose.yml up
#docker compose up -d