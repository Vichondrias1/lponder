@echo off

rem Get repository paths
set REPOS=%1

rem Create full path for the output file
set OUTPUT_FILE=Hooks\updated_file_path.txt

rem Copy content inside tmp to the output file
type %REPOS% > %OUTPUT_FILE%

rem run AI python script that check the codes
python .\Hooks\ai_script.py


