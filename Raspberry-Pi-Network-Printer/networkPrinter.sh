#!/bin/bash

#Update Dependencies
sudo apt update

#install brother printer drivers
sudo apt-get install printer-driver-brlaser

#Restart Cups
sudo service cups restart

