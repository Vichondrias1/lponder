# Table Of Contents 
- [Table Of Contents](#table-of-contents)
- [Copy file from server to NAS using rsync](#copy-file-from-server-to-nas-using-rsync)
- [Fail2ban](#fail2ban)
  - [Install and Setup Fail2ban](#install-and-setup-fail2ban)
  - [Basic Configuration](#basic-configuration)
  - [Enable and Configure SSH jail in jail.local](#enable-and-configure-ssh-jail-in-jaillocal)
  - [Restart Fail2ban](#restart-fail2ban)
  - [Check the ban IPs](#check-the-ban-ips)
  - [Set Time Durations](#set-time-durations)

# Copy file from server to NAS using rsync
See the official documentation of <a href="https://linux.die.net/man/1/rsync" target="_blank">Rsync</a> here.

**Step 1: Connect your server and NAS to tailscale to establish a connect.**

(In this example I use my Deca PC as the Source path and the pi-Lab as the destination path)
![alt text](<../img/tailscale ip.PNG>)

**Step 2: Create a folder on your destination path.**
You need to create a folder on your NAS to point it on the Destination Path and run this commands .

    #Connect to your NAS
    ssh [NAS Username]@[Nas Tailscale IP]

    # Create the directory /home/admin/rsync with superuser privileges.
    sudo mkdir /home/admin/rsync

    # Change the ownership of the /home/admin/rsync directory to user 'admin' and group 'admin'.
    sudo chown admin:admin /home/admin/rsync

    # Add write permissions for the user 'admin' to the /home/admin/rsync directory.
    sudo chmod u+w /home/admin/rsync


**Step 3: Run this command.**



     sudo rsync -av [Source Path] [Destination Username]@[Tailscale Ip]:[Destination Path]

     Example: sudo rsync -av /mnt/c/Users/paqui/Downloads/Rsync-Sample-Data admin@100.70.95.78:/home/admin/Rsync-Folder


# Fail2ban
Fail2Ban is a security tool used to protect servers from brute force attacks and other malicious activities by monitoring log files for suspicious patterns. When it detects repeated failed login attempts or other signs of attacks, it bans the offending IP addresses by updating firewall rules. Common uses include securing SSH, protecting web applications, blocking spam and abuse, and mitigating denial-of-service attacks. It works by monitoring logs, applying filters to identify threats, and executing actions like IP bans to enhance server security.

## Install and Setup Fail2ban

**1. Update the server first.**

    sudo apt update
    sudo apt upgrade

**2. Install Fail2ban**

    sudo apt install fail2ban

## Basic Configuration

Instead of modifying the main configuration file directly, create a local configuration file:

    sudo cp /etc/fail2ban/jail.conf /etc/fail2ban/jail.local

Open the local configuration file for editing:

    sudo nano /etc/fail2ban/jail.local

## Enable and Configure SSH jail in jail.local

Open (jail.local) and find the [sshd] section and set the values to configure the sshd jails:
    
    [sshd]
    enabled = true
    port = ssh
    backend = systemd
    maxretry = 3
    findtime = 300
    bantime = 3600
    ignoreip = 127.0.0.1

- enabled = true: This enables the jail, meaning it will actively monitor and ban IPs based on the specified rules.
- port = ssh: This specifies that the jail will monitor the SSH port. By default, this is port 22, but it can be customized if your SSH service runs on a different port.
- backend = systemd: This tells Fail2Ban to use systemd's journal for log monitoring, which is useful if your system uses systemd (common in modern Linux distributions).
- maxretry = 3: This sets the maximum number of failed login attempts allowed before an IP is banned. In this case, after 3 failed attempts, the IP will be banned.
- findtime = 300: This sets the time window (in seconds) in which the failed login attempts must occur to trigger a ban. Here, if 3 failed attempts occur within 300 seconds (5 minutes), the ban will be triggered.
- bantime = 3600: This sets the duration (in seconds) for which the offending IP will be banned. Here, the ban will last for 3600 seconds (1 hour).
- ignoreip = 127.0.0.1: This specifies IP addresses that should never be banned. In this case, it ensures that the local IP address (127.0.0.1) is never banned, which is useful to avoid locking yourself out.

## Restart Fail2ban
To apply the changes you need to restart fail2ban.

    #Restart fail2ban
    sudo systemctl restart fail2ban

    #Check status
    sudo systemctl status fail2ban


## Check the ban IPs
This command will show the list of banned IP Addresses.

    sudo fail2ban-client status sshd
    
## Set Time Durations

Fail2Ban supports specifying time durations in various units, including minutes, hours, and even days. You can use the following suffixes to specify the time units:

- m for minutes
- h for hours
- d for days

Example

    [sshd]
    enabled = true
    port = ssh
    backend = systemd
    maxretry = 3
    findtime = 5m  # 5 minutes
    bantime = 1h  # 1 hour
    ignoreip = 127.0.0.1

- findtime = 5m sets the time window to 5 minutes.
- bantime = 1h sets the ban duration to 1 hour.





