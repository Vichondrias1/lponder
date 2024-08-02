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
