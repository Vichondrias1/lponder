
![LponderGroup](logo/logo.png)

# Table Of Contents

- [Table Of Contents](#table-of-contents)
- [Welcome to LponderGroup!](#welcome-to-lpondergroup)
- [Default Ports](#default-ports)
- [Uptime Kuma](#uptime-kuma)
- [Pi-Hole](#pi-hole)
- [Dashy](#dashy)
- [Stirling PDF](#stirling-pdf)
- [Nginx](#nginx)
- [Tailscale](#tailscale)
- [Portainer](#portainer)
- [Nginx Proxy Manager](#nginx-proxy-manager)
- [Cups Network Printer (For Brother Printer)](#cups-network-printer-for-brother-printer)

# Welcome to LponderGroup!

Welcome to the Quick Start Guide for Downloading and Running Popular Self-Hosted Applications with Curl Commands!

In this guide, we will cover the essential curl commands to download and run the following applications:

1.  **Uptime Kuma**
2.  **Pi-Hole**
3.  **Dashy**
4.  **Stirling PDF**
5.  **Nginx**
6.  **Tailscale**
7.  **Portainer**
8.  **Nginx Proxy Manager**
9.  **Cups Network Printer**

# Default Ports
The list of ports to access the applications.

|Name|Port Number  |
|--|--|
|Uptime Kuma  | 3001 |
|Pi-Hole  | 84 |
|Dashy  | 4000 |
|Stirling PDF  | 3001 |
|Portainer  | 9000|
|Nginx Proxy Manager  | 82|
|Cups Network Printer  | 631|

# Uptime Kuma
See the official documentation of <a href="https://github.com/louislam/uptime-kuma" target="_blank">Uptime Kuma</a> here.

    curl -sL https://raw.githubusercontent.com/Vichondrias1/lponder/main/Uptime-Kuma/uptimeKuma.sh | sh

# Pi-Hole
See the official documentation of <a href="https://docs.pi-hole.net/" target="_blank">Pi Hole</a> here.

    curl -sL https://raw.githubusercontent.com/Vichondrias1/lponder/main/Pi-Hole/piHole.sh | sh

# Dashy
See the official documentation of <a href="https://dashy.to/docs/" target="_blank">Dashy</a> here.

    curl -sL https://raw.githubusercontent.com/Vichondrias1/lponder/main/Dashy/dashy.sh | sh

# Stirling PDF
See the official documentation of <a href="https://stirlingtools.com/docs/Overview/What%20is%20Stirling-PDF" target="_blank">Stirling PDF</a> here.

    curl -sL https://raw.githubusercontent.com/Vichondrias1/lponder/main/Stirling-PDF/stirlingPDF.sh | sh

# Nginx
See the official documentation of <a href="https://nginx.org/en/docs/" target="_blank">Nginx</a> here.

    curl -sL https://raw.githubusercontent.com/Vichondrias1/lponder/main/Nginx/nginx.sh | sh

# Tailscale
See the official documentation of <a href="https://tailscale.com/kb" target="_blank">Tailscale</a> here.

  1. Use this command to install tailscale on your device.

    curl -fsSL https://tailscale.com/install.sh | sh

  2. Generate the auth key from [tailscale](https://login.tailscale.com/admin/settings/keys) . (Note: Do not include the curly brace)

    export tailscaleAuthkey={Your Authkey Here}

  3. Run the curl command.

    curl -sL https://raw.githubusercontent.com/Vichondrias1/lponder/main/Tailscale/tailscale.sh | sh  

# Portainer
See the official documentation of <a href="https://docs.portainer.io/" target="_blank">Portainer</a> here.

    curl -sL https://raw.githubusercontent.com/Vichondrias1/lponder/main/Portainer/portainer.sh | sh

# Nginx Proxy Manager
See the official documentation of <a href="https://nginxproxymanager.com/guide/" target="_blank">Nginx Proxy Manager</a> here. 

Read the <a href="/Nginx-Proxy-Manager/README.md">Readme.md</a> on how to setup nginx proxy manager on your local machine.

    curl -sL https://raw.githubusercontent.com/Vichondrias1/lponder/main/Nginx-Proxy-Manager/nginxProxyManager.sh | sh

# Cups Network Printer (For Brother Printer)
See the official documentation of <a href="https://www.cups.org/documentation" target="_blank">CUPS</a> here.  (Note that this is only for linux based operating system like Rasberry PI.) Run this curl command and then follow the steps on how to setup the printer to the network.


    1. export IP="Provide Your Ip Range Here"

    2. curl -sL https://raw.githubusercontent.com/Vichondrias1/lponder/main/Raspberry-Pi-Network-Printer/networkPrinter.sh | sh      

**Step 1: Open the Raspberry PI.**

**Step 2: Access the CUPS Web Interface**

Open a web browser and navigate to http://localhost:631.

**Step 3: Log in to the Administration Tab**

Click on the "Administration" tab. You will be prompted to enter your username and password.

**Step 4: Add a Printer**

Click on "Add Printer" and select your local printer from the list. Click "Continue".

**Step 5: Configure Printer Settings**

Enter a desired name and description for your printer. Make sure to check the box next to "Share This Printer". Click "Continue".

**Step 6: Select Printer Make and Model**

Select the make and model of your printer from the list. Click "Add Printer" to complete the setup.

**Step 7: Test the Print Server**

To test the print server, add the printer on another computer and print a test page.

By following these steps, you should be able to successfully add a printer to your CUPS print server and print a test page from another computer.



