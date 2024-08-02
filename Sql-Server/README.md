 # Install MSSQL Server 2022 on Ubuntu 24.04

    curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | sudo gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg

 curl downloads the Microsoft GPG key from the specified URL.
 sudo gpg --dearmor de-armors the key (i.e., extracts the key from the ASCII-armored format).
 The resulting key is saved to /usr/share/keyrings/microsoft-prod.gpg using sudo.

    curl https://packages.microsoft.com/keys/microsoft.asc | sudo tee /etc/apt/trusted.gpg.d/microsoft.asc

Similar to the previous step, this line downloads the Microsoft GPG key again.
sudo tee writes the key to /etc/apt/trusted.gpg.d/microsoft.asc.

    curl -fsSL https://packages.microsoft.com/config/ubuntu/22.04/mssql-server-2022.list | sudo tee /etc/apt/sources.list.d/mssql-server-2022.list

Downloads the Microsoft SQL Server 2022 repository configuration file for Ubuntu 22.04.
sudo tee writes the configuration file to /etc/apt/sources.list.d/mssql-server-2022.list.

# Need to download & install libldap-2.5

    curl -OL http://archive.ubuntu.com/ubuntu/pool/main/o/openldap/libldap-2.5-0_2.5.18+dfsg-0ubuntu0.22.04.1_amd64.deb

Downloads the libldap-2.5-0 package from the Ubuntu archive.

    sudo apt-get install ./libldap-2.5-0_2.5.18+dfsg-0ubuntu0.22.04.1_amd64.deb

Installs the downloaded libldap-2.5-0 package using sudo apt-get.

    sudo apt-get update

Updates the package list to reflect the new Microsoft SQL Server repository.

    sudo apt-get install -y mssql-server

Installs Microsoft SQL Server 2022 using sudo apt-get with the -y flag to assume "yes" to all prompts.

    sudo /opt/mssql/bin/mssql-conf setup

Runs the Microsoft SQL Server configuration setup script. This is the crucial step that sets up the SQL Server instance.

    systemctl status mssql-server --no-pager

Checks the status of the Microsoft SQL Server service using systemctl. The --no-pager flag prevents the output from being paginated.

# Restore Databases Using Shell Command
    #!/bin/bash

    # Set the database credentials and backup file path
    SERVER="localhost"
    USERNAME="SA"
    PASSWORD="Ponder99!"
    BACKUP_PATH="/var/opt/mssql"

    # Create master key
    echo "Creating master key..."
    if  /opt/mssql-tools/bin/sqlcmd -S $SERVER -U $USERNAME -P $PASSWORD -Q "CREATE MASTER KEY ENCRYPTION BY PASSWORD = 'Ponder99!';" ; then
    echo "Master key created successfully."
    else
    echo "Error creating master key. Exiting with error..."
    exit 1
    fi

    #Restore the database legacy
    echo "Restoring database Legacy..."
    if  /opt/mssql-tools/bin/sqlcmd  -S $SERVER -U $USERNAME -P $PASSWORD -Q "RESTORE DATABASE legacy FROM DISK = '$BACKUP_PATH/legacy.bak' WITH REPLACE, MOVE 'Legacy' TO '/var/opt/mssql/data/Legacy_Primary.mdf', MOVE 'Legacy_log' TO '/var/opt/mssql/data/Legacy_Primary.ldf'" ; then
    echo "Database restored successfully."
    else
    echo "Error restoring database. Exiting with error..."
    exit 1
    fi

    # Restore the database hometown
    echo "Restoring database Hometown..."
    if  /opt/mssql-tools/bin/sqlcmd  -S $SERVER -U $USERNAME -P $PASSWORD -Q "RESTORE DATABASE Hometown FROM DISK = '$BACKUP_PATH/hometown.bak' WITH REPLACE, MOVE 'V022U03CPU_Data' TO '/var/opt/mssql/data/Hometown.mdf', MOVE 'V022U03CPU_log' TO '/var/opt/mssql/data/Hometown_l.ldf'" ; then
    echo "Database restored successfully."
    else
    echo "Error restoring database. Exiting with error..."
    exit 1
    fi

    # Restore the database JDE
    echo "Restoring database JDE_PRODUCTION..."
    if  /opt/mssql-tools/bin/sqlcmd  -S $SERVER -U $USERNAME -P $PASSWORD -Q "RESTORE DATABASE JDE_PRODUCTION FROM DISK = '$BACKUP_PATH/JDE-DB01_JDE_PRODUCTION_FULL_20240706_211726.bak' WITH REPLACE, MOVE 'JDE_PRODUCTION_Data' TO '/var/opt/mssql/data/JDE_PRODUCTION_base_Data.MDF', MOVE 'JDE_PRODUCTION_log' TO '/var/opt/mssql/data/JDE_PRODLog.ldf'" ; then
    echo "Database restored successfully."
    else
    echo "Error restoring database. Exiting with error..."
    exit 1
    fi

    #Restore the database inspire
    echo "Restoring database Inspire..."
    if  /opt/mssql-tools/bin/sqlcmd  -S $SERVER -U $USERNAME -P $PASSWORD -Q "RESTORE DATABASE Inspire FROM DISK = '$BACKUP_PATH/inspire.bak' WITH REPLACE, MOVE 'Inspire' TO '/var/opt/mssql/data/Inspire_Primary.mdf', MOVE 'Inspire_log' TO '/var/opt/mssql/data/Inspire_Primary.ldf'" ; then
    echo "Database restored successfully."
    else
    echo "Error restoring database. Exiting with error..."
    exit 1
    fi

    #Restore the database Zeman
    echo "Restoring database Zeman..."
    if  /opt/mssql-tools/bin/sqlcmd  -S $SERVER -U $USERNAME -P $PASSWORD -Q "RESTORE DATABASE Zeman FROM DISK = '$BACKUP_PATH/Zeman.bak' WITH REPLACE, MOVE 'Zeman' TO '/var/opt/mssql/data/Zeman_Primary.mdf', MOVE 'Zeman_log' TO '/var/opt/mssql/data/Zeman_Primary.ldf'" ; then
    echo "Database restored successfully."
    else
    echo "Error restoring database. Exiting with error..."
    exit 1
    fi

    #Restore the database ELS
    echo "Restoring database ELS..."
    if  /opt/mssql-tools/bin/sqlcmd  -S $SERVER -U $USERNAME -P $PASSWORD -Q "RESTORE DATABASE ELS FROM DISK = '$BACKUP_PATH/els.bak' WITH REPLACE, MOVE 'ELS' TO '/var/opt/mssql/data/ELS.mdf', MOVE 'ELS_log' TO '/var/opt/mssql/data/ELS_log.LDF'" ; then
    echo "Database restored successfully."
    else
    echo "Error restoring database. Exiting with error..."
    exit 1
    fi

    #Restore the database Lakeshore
    echo "Restoring database Lakeshore..."
    if  /opt/mssql-tools/bin/sqlcmd  -S $SERVER -U $USERNAME -P $PASSWORD -Q "RESTORE DATABASE Lakeshore FROM DISK = '$BACKUP_PATH/Lakeshore.bak' WITH REPLACE, MOVE 'Lakeshore' TO '/var/opt/mssql/data/Lakeshore_Primary.mdf', MOVE 'Lakeshore_log' TO '/var/opt/mssql/data/Lakeshore_Primary.ldf'" ; then
    echo "Database restored successfully."
    else
    echo "Error restoring database. Exiting with error..."
    exit 1
    fi

    # Create TDE certificate from file
    echo "Creating TDE certificate from file..."
    if  /opt/mssql-tools/bin/sqlcmd -S $SERVER -U $USERNAME -P $PASSWORD -Q "CREATE CERTIFICATE TDECert FROM FILE = '$BACKUP_PATH/Cambio2024TDECert 2.cert' WITH PRIVATE KEY (FILE = '$BACKUP_PATH/Cambio2024 2.key', DECRYPTION BY PASSWORD = 'YmPuWujnJqAKzbbgNi0K') ;" ; then
    echo "TDE certificate created successfully."
    else
    echo "Error creating TDE certificate. Exiting with error..."
    exit 1
    fi

    #Restore databse Cambio
    echo "Restoring database Cambio..."
    if  /opt/mssql-tools/bin/sqlcmd  -S $SERVER -U $USERNAME -P $PASSWORD -Q "RESTORE DATABASE Cambio FROM DISK = '$BACKUP_PATH/Cambio.bak' WITH REPLACE, MOVE 'Cambio' TO '/var/opt/mssql/data/Cambio_Primary.mdf', MOVE 'Cambio_log' TO '/var/opt/mssql/data/Cambio_Primary.ldf'" ; then
    echo "Database restored successfully."
    else
    echo "Error restoring database. Exiting with error..."
    exit 1
    fi

    echo "All databases restored successfully. Exiting..."
    exit 0

# Overview

- This script is designed to restore multiple databases from backup files using SQL Server's command-line tool, sqlcmd. The script creates a master key, restores six databases (Legacy, Hometown, JDE_PRODUCTION, Inspire, Zeman, ELS, Lakeshore, and Cambio), and creates a TDE certificate from a file.

# Prerequisites

- SQL Server installation with sqlcmd tool available
- Backup files for each database in the specified backup path
- Necessary permissions to restore databases and create certificates

# Script Configuration

The script uses the following variables, which can be modified as needed:

- **SERVER** the server name or IP address (default: localhost)
- **USERNAME** the username to connect to the server (default: SA)
- **PASSWORD** the password to connect to the server (default: Ponder99!)
- **BACKUP_PATH** the path where the backup files are located (default: /var/opt/mssql)

# Script Execution

- Create a master key using the specified password.
- Restore each database from its corresponding backup file, replacing any existing database with the same name. The script moves the database files to their respective locations.
- Create a TDE certificate from a file, using the specified private key and decryption password.

# Error Handling

- The script checks for errors after each database restoration and certificate creation step. If an error occurs, the script exits with an error message and a non-zero exit code.

# Exit Codes

- **0**: All databases restored successfully
- **1**: Error occurred during script execution (check error messages for details)

# Security Considerations

- Ensure that the script is executed with the necessary permissions to restore databases and create certificates.
- Use secure passwords and keep them confidential.
- Verify the integrity of the backup files and certificates before executing the script.

# Troubleshooting

- Check the script output for error messages and debug information.
- Verify that the backup files and certificates are valid and accessible.
- Consult the SQL Server documentation for troubleshooting sqlcmd errors.

# Maintenance and Updates

- Periodically review and update the script to ensure it remains compatible with changes to the database schema or SQL Server versions.
- Test the script regularly to ensure it continues to function correctly.

By following this documentation, you should be able to successfully execute the script and restore your databases from backup files.