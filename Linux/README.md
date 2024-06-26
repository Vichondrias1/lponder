# Table Of Contents
- [Table Of Contents](#table-of-contents)
- [Copy file from server to NAS using rsync](#copy-file-from-server-to-nas-using-rsync)

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