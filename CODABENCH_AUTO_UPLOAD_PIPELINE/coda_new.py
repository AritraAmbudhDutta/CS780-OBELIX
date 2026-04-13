"""
CodaBench Automated File Upload Script
Automatically uploads files one at a time to CodaBench submission system
Competition: Reinforcement Learning Challenge! CS780
"""

import os
import time
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.common.keys import Keys
import getpass

# ============================================================================
# CONFIGURATION - MODIFY THESE VALUES
# ============================================================================

# Path to the folder containing your submission files
SUBMISSION_FOLDER = r"C:\Users\neeld\Downloads\CS780_PROJECT\SUBMISSIONS\approach_1"

# CodaBench credentials (leave empty to be prompted)
CODABENCH_USERNAME = ""  # Your CodaBench username/email
CODABENCH_PASSWORD = ""  # Your CodaBench password (leave empty for secure input)

# Competition URLs
CODABENCH_LOGIN_URL = "https://www.codabench.org/accounts/login/"
COMPETITION_ID = "14572"
COMPETITION_URL = f"https://www.codabench.org/competitions/{COMPETITION_ID}/"
SUBMISSION_TAB_URL = f"https://www.codabench.org/competitions/{COMPETITION_ID}/#/participate-tab"

# File extensions to upload (modify based on what you're submitting)
VALID_EXTENSIONS = ['.zip']  

# Time to wait between uploads (seconds) - adjust if needed
WAIT_BETWEEN_UPLOADS = 5

# Maximum time to wait for elements to load (seconds)
TIMEOUT = 20

# ============================================================================
# SCRIPT
# ============================================================================

def setup_driver():
    """Initialize and return Chrome WebDriver"""
    options = webdriver.ChromeOptions()
    
    # Keep browser open if you want to see what's happening
    # options.add_argument('--headless')  # Uncomment to run in background
    
    # Use existing Chrome profile (to maintain login session)
    # Uncomment and modify path if you want to use your existing Chrome profile:
    # options.add_argument("user-data-dir=/path/to/chrome/profile")
    
    driver = webdriver.Chrome(options=options)
    driver.maximize_window()
    return driver


def get_submission_files(folder_path):
    """Get list of files to submit from the folder"""
    folder = Path(folder_path)
    
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    files = []
    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() in VALID_EXTENSIONS:
            files.append(file)
    
    files.sort()  # Sort alphabetically
    return files


def wait_for_element(driver, by, value, timeout=TIMEOUT):
    """Wait for an element to be present and return it"""
    try:
        element = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((by, value))
        )
        return element
    except TimeoutException:
        print(f"Timeout waiting for element: {by}={value}")
        return None


def wait_for_clickable(driver, by, value, timeout=TIMEOUT):
    """Wait for an element to be clickable and return it"""
    try:
        element = WebDriverWait(driver, timeout).until(
            EC.element_to_be_clickable((by, value))
        )
        return element
    except TimeoutException:
        print(f"Timeout waiting for clickable element: {by}={value}")
        return None


def login_to_codabench(driver, username, password):
    """Log in to CodaBench"""
    print("\n" + "="*60)
    print("LOGGING IN TO CODABENCH")
    print("="*60)
    
    try:
        # Navigate to login page
        print(f"Navigating to login page...")
        driver.get(CODABENCH_LOGIN_URL)
        time.sleep(2)
        
        # Find username/email field
        username_selectors = [
            (By.ID, "id_username"),
            (By.NAME, "username"),
            (By.CSS_SELECTOR, "input[type='text']"),
            (By.CSS_SELECTOR, "input[type='email']"),
            (By.XPATH, "//input[@placeholder='Username' or @placeholder='Email']"),
        ]
        
        username_field = None
        for by, selector in username_selectors:
            try:
                username_field = wait_for_element(driver, by, selector, timeout=5)
                if username_field:
                    print(f"Found username field")
                    break
            except:
                continue
        
        if not username_field:
            print("ERROR: Could not find username/email field")
            return False
        
        # Find password field
        password_field = wait_for_element(driver, By.CSS_SELECTOR, "input[type='password']")
        if not password_field:
            print("ERROR: Could not find password field")
            return False
        
        # Enter credentials
        print("Entering credentials...")
        username_field.clear()
        username_field.send_keys(username)
        time.sleep(0.5)
        
        password_field.clear()
        password_field.send_keys(password)
        time.sleep(0.5)
        
        # Find and click login button
        login_button_selectors = [
            (By.CSS_SELECTOR, "button[type='submit']"),
            (By.XPATH, "//button[contains(text(), 'Log') or contains(text(), 'Sign')]"),
            (By.ID, "submit-id-submit"),
            (By.NAME, "submit"),
        ]
        
        login_button = None
        for by, selector in login_button_selectors:
            try:
                login_button = wait_for_clickable(driver, by, selector, timeout=3)
                if login_button:
                    break
            except:
                continue
        
        if not login_button:
            # Try submitting form with Enter key
            print("Submitting with Enter key...")
            password_field.send_keys(Keys.RETURN)
        else:
            print("Clicking login button...")
            login_button.click()
        
        # Wait for login to complete
        time.sleep(3)
        
        # Check if login was successful
        if "login" in driver.current_url.lower():
            print("ERROR: Login failed - still on login page")
            print("Please check your credentials")
            return False
        
        print("✓ Login successful!")
        return True
        
    except Exception as e:
        print(f"ERROR during login: {str(e)}")
        return False


def find_upload_button(driver):
    """Find the upload button near 'Submission upload' text"""
    print("Looking for upload button...")
    
    # Look for text "Submission upload" first to locate the area
    try:
        section = driver.find_element(By.XPATH, "//*[contains(text(), 'Submission upload')]")
        print("✓ Found 'Submission upload' section")
    except:
        print("Note: 'Submission upload' text not found, searching for button anyway")
    
    # Try different selectors for the upload button
    button_selectors = [
        # Look near "Submission upload" text
        (By.XPATH, "//*[contains(text(), 'Submission upload')]/following::button[1]"),
        (By.XPATH, "//*[contains(text(), 'Submission upload')]/following::input[@type='file']/following::button[1]"),
        
        # Look near "Submit as" text  
        (By.XPATH, "//*[contains(text(), 'Submit as')]/following::button[1]"),
        (By.XPATH, "//*[contains(text(), 'Submit as')]/preceding::button[1]"),
        
        # General button searches
        (By.XPATH, "//button[contains(text(), 'Choose') or contains(text(), 'Browse') or contains(text(), 'Select')]"),
        (By.XPATH, "//button[contains(@class, 'file') or contains(@class, 'upload')]"),
        (By.CSS_SELECTOR, "button.btn-primary"),
        (By.CSS_SELECTOR, "label[for*='file']"),  # Sometimes it's a label acting as button
    ]
    
    for by, selector in button_selectors:
        try:
            button = wait_for_clickable(driver, by, selector, timeout=3)
            if button and button.is_displayed():
                print(f"✓ Found upload button")
                return button
        except:
            continue
    
    return None


def upload_file(driver, file_path, upload_number, total_files):
    """Upload a single file to CodaBench"""
    print(f"\n{'='*60}")
    print(f"Upload {upload_number}/{total_files}: {file_path.name}")
    print(f"{'='*60}")
    
    try:
        # Find the upload button
        upload_button = find_upload_button(driver)
        
        if not upload_button:
            print("\n⚠ Could not find upload button automatically.")
            print("Looking for file input directly...")
        
        # Find file input - it might be hidden behind the button
        file_input = None
        try:
            file_inputs = driver.find_elements(By.CSS_SELECTOR, "input[type='file']")
            if file_inputs:
                file_input = file_inputs[0]  # Use first file input found
                print(f"✓ Found file input element")
        except:
            pass
        
        if not file_input:
            print("\n❌ ERROR: Could not find file input element")
            print("Please check if you're on the correct page")
            return False
        
        # If we found the button, click it (might open a styled file picker)
        if upload_button:
            try:
                upload_button.click()
                print("Clicked upload button")
                time.sleep(1)
            except:
                pass  # Button click might fail, but we can still use file input directly
        
        # Send file path to the input
        print(f"Uploading: {file_path.name}")
        file_input.send_keys(str(file_path.absolute()))
        print(f"✓ File selected")
        time.sleep(2)
        
        # Look for submit/confirm button if it appears
        try:
            confirm_selectors = [
                (By.XPATH, "//button[contains(text(), 'Submit') or contains(text(), 'Upload') or contains(text(), 'Confirm')]"),
                (By.CSS_SELECTOR, "button[type='submit']"),
            ]
            
            for by, selector in confirm_selectors:
                try:
                    buttons = driver.find_elements(by, selector)
                    for btn in buttons:
                        if btn.is_displayed() and btn.is_enabled():
                            btn.click()
                            print("✓ Clicked submit button")
                            time.sleep(1)
                            break
                except:
                    continue
        except:
            pass
        
        # Wait for upload to process
        print(f"Processing upload ({WAIT_BETWEEN_UPLOADS}s)...")
        time.sleep(WAIT_BETWEEN_UPLOADS)
        
        # Check for success/error messages
        success = False
        try:
            # Look for success
            success_elements = driver.find_elements(By.XPATH, "//*[contains(text(), 'success') or contains(text(), 'Success') or contains(text(), 'submitted')]")
            for elem in success_elements:
                if elem.is_displayed():
                    print(f"✓ Upload successful!")
                    success = True
                    break
            
            # Look for errors if no success found
            if not success:
                error_elements = driver.find_elements(By.XPATH, "//*[contains(text(), 'error') or contains(text(), 'Error') or contains(text(), 'failed')]")
                for elem in error_elements:
                    if elem.is_displayed():
                        print(f"⚠ Error message: {elem.text[:100]}")
                        return False
        except:
            pass
        
        # If no explicit success/error message, assume success
        if not success:
            print(f"✓ Upload completed for {file_path.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False


def main():
    """Main execution function"""
    print("="*60)
    print("CodaBench Automated File Upload Script")
    print("Competition: Reinforcement Learning Challenge! CS780")
    print("="*60)
    
    # Get credentials
    username = CODABENCH_USERNAME
    password = CODABENCH_PASSWORD
    
    if not username:
        print("\nPlease enter your CodaBench credentials:")
        username = input("Username/Email: ").strip()
    
    if not password:
        password = getpass.getpass("Password: ")
    
    if not username or not password:
        print("ERROR: Username and password are required")
        return
    
    # Get files to upload
    try:
        files = get_submission_files(SUBMISSION_FOLDER)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nPlease update SUBMISSION_FOLDER in the script configuration")
        return
    
    if not files:
        print(f"\nNo files found in {SUBMISSION_FOLDER}")
        print(f"Looking for files with extensions: {VALID_EXTENSIONS}")
        return
    
    print(f"\nFound {len(files)} file(s) to upload:")
    for i, file in enumerate(files, 1):
        print(f"  {i}. {file.name}")
    
    # Confirm before proceeding
    print(f"\nThis will upload ALL {len(files)} files automatically.")
    response = input("Proceed? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled")
        return
    
    # Initialize browser
    print("\nInitializing browser...")
    driver = setup_driver()
    
    try:
        # Login to CodaBench
        login_success = login_to_codabench(driver, username, password)
        
        if not login_success:
            print("\nLogin failed. Please check your credentials and try again.")
            input("Press ENTER to close browser...")
            return
        
        # Navigate to submission tab
        print(f"\nNavigating to submission tab...")
        driver.get(SUBMISSION_TAB_URL)
        time.sleep(3)
        
        print("\n" + "="*60)
        print("STARTING AUTOMATED UPLOAD")
        print("="*60)
        print(f"Uploading {len(files)} files...")
        print("The script will handle all uploads automatically.")
        print("DO NOT close the browser window.")
        print("\n" + "="*60)
        
        # Upload each file automatically
        successful = 0
        failed = 0
        failed_files = []
        
        for i, file in enumerate(files, 1):
            success = upload_file(driver, file, i, len(files))
            
            if success:
                successful += 1
            else:
                failed += 1
                failed_files.append(file.name)
                print(f"\n⚠ Failed: {file.name}")
            
            # Small delay between files (except after last file)
            if i < len(files):
                time.sleep(2)
        
        # Summary
        print("\n" + "="*60)
        print("UPLOAD SUMMARY")
        print("="*60)
        print(f"Total files: {len(files)}")
        print(f"✓ Successful: {successful}")
        print(f"✗ Failed: {failed}")
        
        if failed_files:
            print("\nFailed files:")
            for fname in failed_files:
                print(f"  - {fname}")
        
        print("="*60)
        
        # Navigate back to submission tab to see results
        print("\nRefreshing submission page...")
        driver.get(SUBMISSION_TAB_URL)
        time.sleep(2)
        
        print("\n✓ All uploads completed!")
        print("Check the browser window to see your submissions.")
        input("\nPress ENTER to close browser...")
        
    except KeyboardInterrupt:
        print("\n\nScript interrupted by user (Ctrl+C)")
        print("Stopping uploads...")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        driver.quit()
        print("\nBrowser closed.")


if __name__ == "__main__":
    main()