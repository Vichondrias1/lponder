#!/bin/bash

#Update Dependencies
sudo apt update


#install brother printer drivers
sudo apt-get install printer-driver-brlaser

#Restart Cups
sudo service cups restart

#install ufw
sudo apt install ufw

#Open a terminal on the machine where UFW is configured.
#Enable UFW if it is not already enabled:
sudo ufw enable

#Add the allow rule for your local network:
sudo ufw allow from $IP to any port 631

#Add the deny rule for all other networks:
sudo ufw deny to any port 631

#Add the allow rule for your local network:
sudo ufw allow from $IP to any port 22

#Add the deny rule for all other networks:
sudo ufw deny to any port 22

#Reload UFW to apply the new rules:
sudo ufw reload

#Check the status of UFW to ensure the rules are correctly applied:
sudo ufw status

