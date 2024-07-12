import sys
import os
import json

def has_issue_in_json(jsonDataPath):


  try:
    with open(jsonDataPath, 'r') as f:
      data = json.load(f)
  except FileNotFoundError:
    return 0
  except json.JSONDecodeError:
    return 0

  # Check if any object has "hasAnIssue": 1 using list comprehension
  has_issue = any(item.get("hasAnIssue", 0) == 1 for item in data)
  
  return int(has_issue)


def read_and_print_file():

    errorOutputName = 'Hooks/issues_output.txt'
    updatedFilePath = 'Hooks/updated_file_path.txt'
    jsonDataPath = 'Hooks/ai_response.json'

    has_issue = has_issue_in_json(jsonDataPath)

    if has_issue == 1:

         # Read the contents of the file
        with open(updatedFilePath, 'r') as filepath:
            message = filepath.read()

        # Print the message to both standard output and standard error
        print(message, file=sys.stderr)

        # Read the contents of the file
        with open(errorOutputName, 'r') as file:
            message = file.read()

        # Print the message to both standard output and standard error
        print(message, file=sys.stderr)


    os.remove(errorOutputName)
    os.remove(updatedFilePath)
    os.remove(jsonDataPath)

    # Exit with status 1
    sys.exit(has_issue)


# display the error message in SVN
read_and_print_file()
