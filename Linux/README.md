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
- [Scrape ELS Website For PDF's and HTML Files](#scrape-els-website-for-pdfs-and-html-files)

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


# Scrape ELS Website For PDF's and HTML Files

        import os
        import requests
        from bs4 import BeautifulSoup
        import markdownify
        from datetime import datetime
        import json
        from pdfminer.high_level import extract_text

        # Function to extract text from a PDF file using pdfminer
        def extract_text_from_pdf(pdf_path):
            return extract_text(pdf_path)

        # Send an HTTP request to the web page and get the HTML response
        url = "https://equitylifestyle.gcs-web.com/news?af7c0978_year%5Bvalue%5D=_none"  # Replace with the actual URL
        response = requests.get(url)
        html = response.content

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')

        # Find all table rows (tr) in the HTML content
        table_rows = soup.select('table.tabcon tbody tr')

        # Display the total number of rows being scraped
        print(f"Total rows being scraped: {len(table_rows)}")

        # Loop through each row and extract the links
        links = []
        for row in table_rows:
            # Find the 'td' element containing the link
            link_element = row.find('td', {'id': 'news-title'})
            date_element = row.find('td', {'id': 'news-date'})
            
            if link_element and link_element.a and date_element:
                # Extract the link text (title)
                title_text = link_element.text.strip()
                
                # Extract the date text
                date_text = date_element.text.strip()
                
                # Extract the link URL and type
                link_url = link_element.a['href']
                link_type = link_element.a.get('type', '')

                # Construct the full URL
                full_link = requests.compat.urljoin(url, link_url)
                links.append((title_text, full_link, link_type, date_text))

        # Create directories for PDF and markdown files if they don't exist
        pdf_folder = "pdf"
        md_folder = "markdown"
        if not os.path.exists(pdf_folder):
            os.makedirs(pdf_folder)
        if not os.path.exists(md_folder):
            os.makedirs(md_folder)

        # Initialize counters for downloads
        pdf_count = 0
        md_count = 0
        file_counter = 1  # Counter to ensure unique filenames

        # List to hold JSON data
        json_data = []

        # Download each link
        for title, link, link_type, date in links:
            response = requests.get(link)
            # Generate a valid filename from the title
            valid_title = "".join(c if c.isalnum() or c in " .-_" else " " for c in title)
            
            # Generate the current timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Append the counter to the filename to ensure uniqueness
            file_id = f"{file_counter:03d}"  # Zero-padded counter
            file_counter += 1
            
            file_path = ""
            content = ""
            
            if link_type in ['application/pdf', 'application/octet-stream']:
                # Save the PDF file with the timestamp and counter
                file_path = os.path.join(pdf_folder, f"{valid_title}_{timestamp}_{file_id}.pdf")
                with open(file_path, 'wb') as file:
                    file.write(response.content)
                pdf_count += 1
                print(f"Downloaded PDF: {title}")
                
                # Extract text from the PDF file
                content = extract_text_from_pdf(file_path)
                
            else:
                # Parse the HTML content of the page
                page_soup = BeautifulSoup(response.content, 'html.parser')
                body_content = page_soup.find('div', {'id': 'content-area', 'class': 'container'})
                
                if body_content:
                    # Convert the body content to markdown
                    markdown_content = markdownify.markdownify(str(body_content), heading_style="ATX")
                    
                    # Save the markdown file with the timestamp and counter
                    file_path = os.path.join(md_folder, f"{valid_title}_{timestamp}_{file_id}.md")
                    with open(file_path, 'w', encoding='utf-8') as file:
                        file.write(markdown_content)
                    md_count += 1
                    print(f"Downloaded and converted to Markdown: {title}")
                    
                    # Use the markdown content as the file content
                    content = markdown_content
            
            # Add to JSON data list
            json_data.append({
                "title": title,
                "content": content,
                "date": date
            })

        # Save JSON data to a file
        json_file_path = "downloaded_data.json"
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(json_data, json_file, ensure_ascii=False, indent=4)

        # Display the download counts
        total_count = pdf_count + md_count
        print(f"Total PDFs downloaded: {pdf_count}")
        print(f"Total Markdown files downloaded: {md_count}")
        print(f"Total files downloaded: {total_count}")
        print(f"JSON data saved to {json_file_path}")


Here’s a line-by-line explanation:

1. Import necessary libraries: These libraries are used for various tasks such as web scraping, PDF text extraction, markdown conversion, handling dates, and JSON operations.

        pip install request 
        pip install beautifulsoup4
        pip install markdownify
        pip install pdfminer
        pip install pdfminer.six

2. Define a function to extract text from a PDF file using pdfminer. This function takes a PDF file path as input and returns the extracted text.
3. Send an HTTP GET request to the specified URL and get the HTML content of the page.
4. Parse the HTML content using BeautifulSoup to create a BeautifulSoup object (soup) which allows us to navigate and search the HTML structure.
5. Find all table rows (tr) within the table in the parsed HTML. This is done using the select method with a CSS selector.
6. Print the total number of rows that are being scraped to provide feedback on how many rows were found.
7. Loop through each row in the table to extract the relevant links:
    - Find the 'td' element containing the link (news-title) and the date (news-date).
    - Check if the link and date elements exist and if so, extract their text content and the link URL.
    - Construct the full URL of the link using requests.compat.urljoin to handle relative URLs.
    - Append the extracted data (title, full link, link type, date) to the links list.
8. Create directories for PDF and markdown files if they do not already exist using os.makedirs.
9. Initialize counters for tracking the number of PDF and markdown files downloaded and a counter to ensure unique filenames.
10. Create a list to hold JSON data for all the downloaded files.
11. Loop through each link to download the content:
    - Send an HTTP GET request to the link URL.
    - Generate a valid filename from the title by removing or replacing invalid characters.
    - Generate a timestamp for the current date and time.
    - Create a unique file ID using the counter.
    - Initialize variables for the file path and content.
    - Check the link type to determine if it's a PDF or HTML content:
        - For PDF files:
            - Save the PDF file with the generated filename.
            - Extract text from the PDF using the extract_text_from_pdf function.
            - Increment the PDF counter and print a message.
        - For HTML content:
            - Parse the HTML content using BeautifulSoup.
            - Find the main content area in the parsed HTML.
            - Convert the content to markdown using markdownify.
            - Save the markdown file with the generated filename.
            - Increment the markdown counter and print a message.
            - Use the markdown content as the file content.
    - Append the extracted data (title, content, date) to the JSON data list.
12. Save the JSON data to a file (downloaded_data.json) using json.dump.
13. Print the download counts for PDFs, markdown files, and the total number of files. Also, print the path to the saved JSON file.

This code automates the process of scraping a news page, downloading PDF or HTML content, converting HTML to markdown, extracting text from PDFs, and saving all this information in a structured JSON format.



