#!/bin/bash

sudo mkdir uptime-kuma && cd uptime-kuma

curl -O https://raw.githubusercontent.com/Vichondrias1/lponder/main/docker-compose.yml

docker compose up -d