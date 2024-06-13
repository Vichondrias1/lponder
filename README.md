 ![LponderGroup](logo/logo.png)
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
See the official documentation of <a href="https://www.cups.org/documentation" target="_blank">CUPS</a> here.  (Note that this is only for linux based operating system like Rasberry PI.)


    curl -sL https://raw.githubusercontent.com/Vichondrias1/lponder/main/Raspberry-Pi-Network-Printer/networkPrinter.sh | sh    



