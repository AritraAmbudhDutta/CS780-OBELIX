# CodaBench Automated File Upload Script

This script automates the process of uploading multiple files to CodaBench competitions, one at a time.

**Competition:** Reinforcement Learning Challenge! CS780  
**Competition ID:** 14572  
**URL:** https://www.codabench.org/competitions/14572/

## Prerequisites

- Python 3.7 or higher
- Google Chrome browser installed
- CodaBench account with access to the competition
- Chrome WebDriver (will be installed automatically)

## Setup Instructions

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install selenium webdriver-manager
```

### 2. Configure the Script

Open `codabench_auto_upload.py` and modify the configuration section at the top:

```python
# Path to folder with your submission files
SUBMISSION_FOLDER = "/path/to/your/submission/files"

# CodaBench credentials (optional - will prompt if not provided)
CODABENCH_USERNAME = "your_username_or_email"  
CODABENCH_PASSWORD = ""  # Leave empty for secure input prompt

# File types to upload
VALID_EXTENSIONS = ['.zip', '.csv', '.txt', '.py', '.json']
```

**Important Configuration Items:**

- **SUBMISSION_FOLDER**: Full path to the folder containing files you want to upload
  - Example (Windows): `C:/Users/YourName/Documents/submissions`
  - Example (Mac/Linux): `/home/yourname/submissions`

- **CODABENCH_USERNAME**: Your CodaBench username or email
  - You can leave this empty and the script will prompt you
  
- **CODABENCH_PASSWORD**: Your CodaBench password
  - **Security Note**: Leave this EMPTY in the script for secure input
  - The script will prompt you to enter it securely when running

- **VALID_EXTENSIONS**: List of file extensions to upload
  - Modify based on what your competition accepts
  - Script will only upload files with these extensions

### 3. Prepare Your Files

Place all files you want to upload in the folder specified in `SUBMISSION_FOLDER`. The script will:
- Find all files with the specified extensions
- Upload them in alphabetical order
- Skip any files that don't match the extensions

## Usage

### Basic Usage

1. Run the script:
   ```bash
   python codabench_auto_upload.py
   ```

2. When prompted, enter your CodaBench credentials:
   ```
   Username/Email: your_email@example.com
   Password: (hidden input)
   ```

3. The script will:
   - Log in to CodaBench
   - Navigate to your competition
   - Show you the list of files to upload
   - Upload each file one at a time to the participate tab
   - Show a summary when done

### Important Notes

- **Secure Credentials**: Your password is never stored in the script
- **Watch Progress**: The script prints status for each upload
- **Browser Control**: Don't close the browser while script is running
- **Error Recovery**: If an upload fails, you can choose to continue or stop

### Pre-configured Credentials (Optional)

If you want to avoid typing credentials each time, you can add them to the script:

```python
CODABENCH_USERNAME = "your_email@example.com"
CODABENCH_PASSWORD = "your_password"  # Not recommended for security
```

**Security Warning**: Only do this if you're the only user of your computer and the script file is kept secure.

## Troubleshooting

### Login Issues

**"Login failed - still on login page"**
- Double-check your username/email and password
- Make sure your account has access to the competition
- Try logging in manually first to verify credentials

**Two-Factor Authentication (2FA)**
- If your account has 2FA enabled, the script will pause after login
- Complete the 2FA challenge manually in the browser
- The script will continue automatically after you verify

### Upload Issues

**"Could not find submit button"**

The HTML selectors might need adjustment. To fix:

1. Open CodaBench in Chrome manually
2. Go to https://www.codabench.org/competitions/14572/#/participate-tab
3. Right-click the Submit/Upload button → Inspect
4. Find the button's ID, class, or text
5. Update the `button_selectors` list in the script

Example:
```python
button_selectors = [
    (By.ID, "actual-button-id"),  # Replace with actual ID
    (By.CLASS_NAME, "actual-class-name"),  # Or class name
]
```

**"Could not find file input element"**

Similar to above, inspect the file upload input field and update `file_input_selectors`.

### Chrome Driver Issues

If you get ChromeDriver errors:
```bash
pip install --upgrade webdriver-manager
```

## Example Workflow

1. You have 10 submission files: `sub1.zip`, `sub2.zip`, ..., `sub10.zip`
2. Place them in `/home/user/codabench_submissions/`
3. Update `SUBMISSION_FOLDER` in script to point to that folder
4. Run: `python codabench_auto_upload.py`
5. Enter your CodaBench credentials when prompted
6. Script logs in and uploads each file automatically
7. Check your submissions on CodaBench

## Advanced Features

### Run in Background (Headless Mode)

To hide the browser window while uploading:

1. Edit the `setup_driver()` function
2. Uncomment this line:
   ```python
   options.add_argument('--headless')
   ```

**Note**: Headless mode makes debugging harder if something goes wrong.

### Adjust Upload Speed

If you're hitting rate limits or uploads are too fast:

```python
WAIT_BETWEEN_UPLOADS = 10  # Increase to 10 seconds between files
TIMEOUT = 30  # Increase timeout for slow connections
```

## Safety Features

- Prints detailed progress for each file
- Asks for confirmation if upload fails
- Waits between uploads to avoid rate limiting
- Shows summary at the end

## Common Issues

**Q: Script opens browser but doesn't do anything**
A: Make sure the CODABENCH_URL is correct and points to the participants tab

**Q: Upload succeeds but file isn't visible in CodaBench**
A: Check if there's a confirmation step after upload that needs manual clicking

**Q: Script is too fast/slow**
A: Adjust `WAIT_BETWEEN_UPLOADS` value

## Support

If you encounter issues:
1. Check the console output for error messages
2. Verify you're logged in to CodaBench
3. Ensure the competition accepts file submissions
4. Update the HTML selectors if the page structure changed

## License

Free to use and modify as needed.
