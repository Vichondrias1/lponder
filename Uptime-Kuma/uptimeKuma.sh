#!/bin/bash

#tried but doesnt work wtff haha!!!
#sudo mkdir uptime-kuma && cd uptime-kuma

#download the yml file to the current directory
curl -O https://raw.githubusercontent.com/Vichondrias1/lponder/main/docker-compose.yml

#execute the yml file
docker compose up -d