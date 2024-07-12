@echo off
setlocal

rem Create full path for the output file
set PATH=Hooks\updated_file_path.txt

for /f "delims=" %%f in (%PATH%) do (
    echo %%f 1>&2
)

REM python .\Hooks\readErrorMessage.py
REM C:\Users\Liam\AppData\Local\Programs\Python\Python312\python.exe .\Hooks\readErrorMessage.py

