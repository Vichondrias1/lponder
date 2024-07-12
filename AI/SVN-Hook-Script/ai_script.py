import openai
import argparse
import sys
import os
import json
import base64
import requests

def read_file(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None

def analyze_file_with_chat(file_name, file_path):

    open_ai_key = os.getenv("OPENAI_API_KEY")
 
    # Initialize OpenAI client
    openai.api_key = open_ai_key
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {open_ai_key}"
    }

    # Read the file as bytes and encode it to base64
    with open(file_path, 'rb') as file:

        base64_file = base64.b64encode(file.read()).decode('utf-8')


     # Constructing the prompt
    systemPrompt = f"""
    Your the Head Software Tester of LponderGroup, The code is base64 encoded. your job is to decode the base64 and identify issues in the code. 
    Assuming that all dependencies and functions are correct. 


    Here are the list of issue to analyze or check for C# code:
    1. Check if there are any expose or hardcoded api key.
    2. Check if there are any Logic errors (e.g. infinite loops, incorrect conditional statements)
    3. Check if there are any Variable misuse (e.g. undefined variables, incorrect data types)
    4. Check for bugs(e.g missing semicolon, curly braces)

    If you dont find any issue keep it empty.
    If you find any issue write the issues.

    YOU WILL ALWAYS RESPOND IN THIS JSON FORMAT:
    [
        {{
            "fileName": "(name of the file)"
            "filePath": "(Path of the file)"
            "hasAnIssue": (value is 1 or 0)
            "Issue": "(value are the issues and empty if no issue found in string format)"
        }}
    ]   

    REMOVE THE JSON WORD IN THE OUTPUT
        """

    #System Prompt to test success commit
    # systemPrompt = f"""
    # DONT ANALYZE THIS JUST RESPOND WITH THE GIVEN FORMAT

    # YOU WILL ALWAYS RESPOND IN THIS JSON FORMAT:
    # [
    #     {{
    #         "fileName": "(name of the file)"
    #         "filePath": "(Path of the file)"
    #         "hasAnIssue": (value is always 0)
    #         "Issue": "(value is alawys empty string)"
    #     }}
    # ]   

    # REMOVE THE JSON WORD IN THE OUTPUT
    # """    

    userPrompt = f"""
    Analyze the following code, 
    
    Check each of the list if you find any issue and keep it empty if you dont find any issue
    
    Filename: {file_name} File Path: {file_path}
    """    

    payload = {
    "model": "gpt-4o",
    "messages": [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": systemPrompt
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": userPrompt
                },
                {
                    "type": "text",
                    "text": base64_file  # Send base64 encoded content as text
                }
            ]
        }
    ],
    "max_tokens": 1000,
    "temperature": 0.5
    }
 
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response_data = response.json()

    return response_data['choices'][0]['message']['content']
    

def output_to_json_file(content_value):

    try:
        # Parse the string into a JSON object
        json_data = json.loads(content_value)

        json_path = 'Hooks/ai_response.json'

        # Check if the file already exists
        if os.path.exists(json_path):
            # Read the existing data
            with open(json_path, 'r') as json_file:
                try:
                    existing_data = json.load(json_file)
                except json.JSONDecodeError:
                    existing_data = []
        else:
            existing_data = []

        # Append the new data to the existing data
        if isinstance(existing_data, list):
            existing_data.extend(json_data)
        else:
            existing_data = json_data

        # Write the updated data back to the file
        with open(json_path, 'w') as json_file:
            json.dump(existing_data, json_file, indent=4)

        print(f"JSON data has been appended to {json_path}")
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}")
        print("Response content that caused the error:")
        print(content_value)

# Function to read JSON data from file
def read_json_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            try:
                return json.load(file)
            except json.JSONDecodeError:
                return []
    return []

# Function to write issues to text file
def write_issues_to_txt():

    # File path for the JSON data file
    json_file_path = 'Hooks/ai_response.json'

    # File path for the output text file
    output_file_path = 'Hooks/issues_output.txt'

    # Read JSON data from file
    json_data = read_json_file(json_file_path)

    with open(output_file_path, 'w') as txt_file:
        for item in json_data:
            if item.get('hasAnIssue', 0) == 1:  # Only proceed if there is an issue
                file_name = item.get('fileName', 'Unknown file')
                issue = item.get('Issue', 'No issues found')
                txt_file.write(f"----------------[{file_name}]----------------\n")
                txt_file.write(issue + '\n')
                txt_file.write(f"--------------------------------------------\n")
            

def main():
    

    #List of updated paths
    file_path = 'Hooks/updated_file_path.txt'
    
    with open(file_path, 'r') as file:
        for line in file:
            path = line.strip()  # Remove trailing newline character

            file_contents = read_file(path)

            if file_contents: 

                #Gets the filename
                filename = os.path.basename(path)

                #Gets the value being return from openai
                content_value = analyze_file_with_chat(filename, path)

                print(content_value)

                #Append the output from openai and put it into a json file
                output_to_json_file(content_value)      

                # Write the issues into the text file
                write_issues_to_txt()


                    
    # Run Ai SuccessFully
    # 0 success
    # 1 error
    # Used 0 to execute the next pre-commit script
    sys.exit(0)



if __name__ == "__main__":
    main()