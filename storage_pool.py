# -*- coding: utf-8 -*-
import os
import sys
import hashlib
import json
import math
import time
import datetime # Needed for timestamp in input prompt and logging
import base64
import argparse
import platform
import threading
import subprocess # <--- Add this
import shutil     # <--- Add this
import traceback

import shutil



import getpass
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from typing import Optional, Union, Tuple, Dict, Any, List

# --- NEW IMPORTS for Client-Server ---
import requests
import subprocess
import socket
# --- END NEW IMPORTS ---

# --- Rich Table Import (NEW - required for decrypt_main table) ---
try:
    from rich.console import Console
    from rich.prompt import Confirm # For yes/no prompts
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    # Define dummy classes if rich is missing, so the rest of the code doesn't break
    # (although table formatting will be lost in decrypt_main)
    class Console:
        def print(self, *args, **kwargs):
            # Basic print fallback
            print(*args)
        def input(self, prompt: str = "") -> str:
             # Basic input fallback, losing rich formatting for prompt
             # Need to strip potential color codes from prompt if using fallback
             import re
             clean_prompt = re.sub(r'\x1b\[[0-9;]*[mK]', '', prompt) # Basic ANSI strip
             return input(clean_prompt)
    class Table:
        def __init__(self, *args, **kwargs): pass
        def add_column(self, *args, **kwargs): pass
        def add_row(self, *args, **kwargs): pass
    class Confirm: # Dummy confirm
        @staticmethod
        def ask(prompt: str, default: bool = False) -> bool:
            import re
            clean_prompt = re.sub(r'\x1b\[[0-9;]*[mK]', '', prompt) # Basic ANSI strip
            response = input(f"{clean_prompt} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
            if not response: return default
            return response == 'y'
# --- End Rich Table Import ---

# --- PyGithub Integration ---
try:
    # Attempt to import specific exceptions first if available
    try:
        from github import RateLimitExceededException, BadCredentialsException, TwoFactorException, GithubException
    except ImportError:
        # Fallback for older PyGithub versions or incomplete installation
        class GithubException(Exception): pass
        class RateLimitExceededException(GithubException): pass
        class BadCredentialsException(GithubException): pass
        class TwoFactorException(GithubException): pass

    from github import Github, UnknownObjectException, GitRelease, GitReleaseAsset, Repository
    PYGITHUB_AVAILABLE = True
except ImportError:
    PYGITHUB_AVAILABLE = False
    # Define dummy classes if PyGithub is missing
    class Github: pass
    class GithubException(Exception): pass
    class RateLimitExceededException(GithubException): pass
    class BadCredentialsException(GithubException): pass
    class TwoFactorException(GithubException): pass
    class UnknownObjectException(GithubException): pass
    class GitRelease: pass
    class GitReleaseAsset: pass
    class Repository: pass

# --- Requests Integration (Now used for DB Server too) ---
# Check remains the same
try:
    # import requests # Already imported above
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False # Should already be handled, but keep check

# --- Dependency Imports with Error Handling (Existing) ---
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.exceptions import InvalidTag
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    class AESGCM: pass
    class InvalidTag(Exception): pass

try:
    import colorama
    from colorama import Fore, Style, Back
    # Ensure init happens once
    if 'colorama_initialized' not in globals():
        colorama.init(autoreset=True, strip=None if platform.system() == "Windows" else False)
        colorama_initialized = True
except ImportError:
    print("Warning: 'colorama' library not found. Output will not be colored. Install with: pip install colorama")
    class DummyStyle:
        def __getattr__(self, name): return ""
    Fore = DummyStyle(); Back = DummyStyle(); Style = DummyStyle()

try:
    from tqdm import tqdm
    TQDM_BAR_FORMAT = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
except ImportError:
    print("Warning: 'tqdm' library not found. Progress bars will not be shown. Install with: pip install tqdm")
    TQDM_BAR_FORMAT = '{l_bar}{bar}| {n_fmt}/{total_fmt}' # Fallback bar format
    # Keep your existing dummy tqdm class definition here if you have one
    def tqdm(iterable=None, **kwargs):
        if iterable is not None: return iterable
        class DummyTqdm:
            def __init__(self, *args, **kwargs): pass
            def update(self, n=1): pass
            def close(self): pass
            def __enter__(self): return self
            def __exit__(self, exc_type, exc_val, exc_tb): pass
            def set_description_str(self, s): pass
            def set_postfix_str(self, s): pass
        return DummyTqdm()
# --- Timezone Handling Import (NEW) ---
try:
    import pytz
    PYTZ_AVAILABLE = True
    # Define the local timezone based on context
    # Use 'Asia/Kolkata' for IST
    LOCAL_TIMEZONE_STR = 'Asia/Kolkata'
    try:
        LOCAL_TZ = pytz.timezone(LOCAL_TIMEZONE_STR)
    except pytz.UnknownTimeZoneError:
        print(f"Warning: Unknown timezone '{LOCAL_TIMEZONE_STR}'. Falling back to UTC for local time.")
        LOCAL_TZ = pytz.utc
        LOCAL_TIMEZONE_STR = 'UTC (Fallback)'

except ImportError:
    PYTZ_AVAILABLE = False
    LOCAL_TZ = None # Cannot perform timezone conversion
    LOCAL_TIMEZONE_STR = 'Local (pytz not installed)'
    print("Warning: 'pytz' library not found. Local times cannot be displayed. Install with: pip install pytz")
# --- End Timezone Handling Import ---


from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.getcwd() , "config" , "system_config.conf"))
# --- Default Configuration ---
DEFAULT_PART_SIZE_MB = 1800
DEFAULT_HASH_BUFFER_MB = 4
DEFAULT_WORKERS = 2
DEFAULT_DOWNLOAD_WORKERS = 2
MAX_DOWNLOAD_WORKERS = 3
DEFAULT_ENC_OUTPUT_DIR_TEMPLATE = "{}_encrypt_process"
DEFAULT_PROC_OUTPUT_DIR_TEMPLATE = "{}_split_process"
DEFAULT_DECRYPT_TEMP_DIR_TEMPLATE = "_decrypt_{}_{}"
# DB_FILENAME = "file_processor_log.db" # <<< REMOVED - Handled by server >>>
AES_GCM_NONCE_SIZE = 12
AES_GCM_TAG_SIZE = 16
PROCESSOR_VERSION = "1.0.0" # <<< Version Bumped for GitHub Fixes & Full Funcs >>>

# --- DB Server Configuration ---
DB_SERVER_HOST = os.environ.get("DB_SERVER_HOST", "127.0.0.1")
DB_SERVER_PORT = int(os.environ.get("DB_SERVER_PORT", 8034))
DB_SERVER_URL = f"http://{DB_SERVER_HOST}:{DB_SERVER_PORT}"
DB_SERVER_SCRIPT = os.path.join(os.getcwd() , "database" , "db_server.py") # Assumes it's in the same directory

# --- Layer Status Constants ---
LAYER_WAITING = "Waiting"
LAYER_PULLING = "Pulling"
LAYER_VERIFYING_DL = "Verifying SHA (Download)"
LAYER_DOWNLOADED = f"{Fore.GREEN}Pulled{Style.RESET_ALL}"
LAYER_PULL_FAILED = f"{Fore.RED}Pull Failed{Style.RESET_ALL}"
LAYER_SHA_MISMATCH_DL = f"{Fore.RED}SHA Mismatch (Download){Style.RESET_ALL}"
LAYER_WAITING_EXTRACT = "Queued (Extract)"
LAYER_EXTRACTING = f"{Fore.YELLOW}Extracting{Style.RESET_ALL}"
LAYER_EXTRACTED = f"{Fore.GREEN}Extracted{Style.RESET_ALL}"
LAYER_EXTRACT_FAILED = f"{Fore.RED}Extraction Failed{Style.RESET_ALL}"
INPUT_SEPARATOR = f"{Fore.YELLOW}{'-'*20} INPUT REQUIRED {Style.RESET_ALL}{Fore.YELLOW}{'-'*20}{Style.RESET_ALL}"

import re # Ensure 're' is imported at the top of the file

def sanitize_release_version_tag(tag_name: str) -> str:
    """
    Sanitizes a proposed tag name to be compliant with GitHub release tag rules.
    Removes or replaces potentially invalid characters.

    Args:
        tag_name: The proposed tag name string.

    Returns:
        A sanitized version of the tag name.
    """
    if not tag_name:
        return "default_sanitized_tag" # Return a default if input is empty

    # Define invalid characters based on common Git ref/tag rules
    # Ref: https://git-scm.com/docs/git-check-ref-format
    # Avoid: ASCII control chars, space, ~, ^, :, ?, *, [, \, consecutive dots .., @{, / at end, .lock at end
    # We will be conservative and replace most symbols with hyphens.
    
    # Replace common problematic characters with hyphen
    sanitized = re.sub(r'[ \t\n\r\f\v~^:?*\[\\@\{]+', '-', tag_name)
    
    # Replace multiple consecutive hyphens with a single hyphen
    sanitized = re.sub(r'-{2,}', '-', sanitized)
    
    # Remove leading/trailing hyphens and dots
    sanitized = sanitized.strip('-.')
    
    # Remove problematic sequences like .. and /.
    sanitized = sanitized.replace('..', '-') # Replace double dots
    sanitized = sanitized.replace('/.', '/-') # Replace /.
    
    # Remove trailing / and .lock
    if sanitized.endswith('/'):
        sanitized = sanitized[:-1]
    if sanitized.endswith('.lock'):
        sanitized = sanitized[:-5]
        
    # If sanitization results in an empty string, return a default
    if not sanitized:
        return "sanitized_tag_fallback"

    # GitHub tags often have a length limit (though not strictly enforced everywhere)
    # Truncate reasonably if needed (e.g., 200 chars)
    max_len = 200
    if len(sanitized) > max_len:
        print_log("WARN", f"Sanitized tag exceeded {max_len} chars, truncating.")
        sanitized = sanitized[:max_len].strip('-.') # Re-strip after potential truncate

    # Ensure it doesn't end up empty after final strip
    if not sanitized:
        return "final_sanitized_tag_fallback"

    print_log("DEBUG", f"Sanitized tag '{tag_name}' -> '{sanitized}'")
    return sanitized

# --- Logging Helpers ---
log_lock = threading.Lock()

def print_log(level, message, part_num=None, clear_line=False):
    """Prints formatted log messages thread-safely."""
    with log_lock:
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        thread_name = threading.current_thread().name
        part_str = f"[Layer {part_num}]" if part_num is not None else "" # Changed Part -> Layer

        log_prefix_map = {
            "INFO":    f"{Fore.CYAN}[{timestamp}][{thread_name}][INFO]{part_str}{Style.RESET_ALL}",
            "WARN":    f"{Fore.YELLOW}{Style.BRIGHT}[{timestamp}][{thread_name}][WARN]{part_str}{Style.RESET_ALL}",
            "ERROR":   f"{Fore.RED}{Style.BRIGHT}[{timestamp}][{thread_name}][ERROR]{part_str}{Style.RESET_ALL}",
            "SUCCESS": f"{Fore.GREEN}{Style.BRIGHT}[{timestamp}][{thread_name}][SUCCESS]{part_str}{Style.RESET_ALL}",
            "STEP":    f"\n{Fore.MAGENTA}{Style.BRIGHT}>>> [{timestamp}][{thread_name}] {message} <<<{Style.RESET_ALL}",
            "DEBUG":   f"{Fore.WHITE}[{timestamp}][{thread_name}][DEBUG]{part_str}{Style.RESET_ALL}",
            "INPUT":   f"{Fore.YELLOW}> [{timestamp}][{thread_name}][INPUT]{part_str}{Style.RESET_ALL} {message}",
            "DB":      f"{Fore.BLUE}{Style.BRIGHT}[{timestamp}][{thread_name}][DB_CLIENT]{part_str}{Style.RESET_ALL}", # Changed prefix
        }
        prefix = log_prefix_map.get(level, f"[{timestamp}][{thread_name}][{level}]{part_str}")

        try:
            terminal_width = shutil.get_terminal_size().columns
        except OSError:
            terminal_width = 80 # Fallback if terminal size can't be determined

        if level == "STEP":
            print(prefix)
        elif level == "INPUT":
            print(f"{prefix}", end="")
            sys.stdout.flush()
        else:
            padding = " " * terminal_width
            line_content = f"{prefix} {message}"
            print(f"\r{padding}\r{line_content[:terminal_width]}".ljust(terminal_width))


# --- Animation Helper ---
def upload_animation(stop_event: threading.Event, asset_name: str):
    """Displays a simple animation during upload."""
    animation_chars = ['.  ', '.. ', '...', ' ..', '  .']
    idx = 0
    try:
        terminal_width = shutil.get_terminal_size().columns
    except OSError:
        terminal_width = 80 # Fallback

    while not stop_event.is_set():
        text = f"\r{Fore.YELLOW}Uploading asset '{asset_name}' [{animation_chars[idx % len(animation_chars)]}]{Style.RESET_ALL}"
        print(text.ljust(terminal_width), end="")
        idx += 1
        time.sleep(0.2)
    print("\r".ljust(terminal_width), end="")
    sys.stdout.flush()


# --- Helper Functions ---

def calculate_sha256(filepath, total_size, buffer_size, desc_prefix="Hashing", position=None, leave=False):
    """Calculates SHA256 of a file with progress bar."""
    sha256_hash = hashlib.sha256()
    pbar = None
    try:
        with open(filepath, 'rb') as f:
            pbar = tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024,
                        desc=f"{Fore.BLUE}{desc_prefix}{Style.RESET_ALL}",
                        bar_format=TQDM_BAR_FORMAT,
                        ncols=90, leave=leave, position=position if position is not None else 0)
            while True:
                data = f.read(buffer_size)
                if not data: break
                sha256_hash.update(data)
                pbar.update(len(data))
            return sha256_hash.hexdigest()
    except FileNotFoundError:
        print_log("ERROR", f"File not found for hashing: {filepath}")
        return None
    except Exception as e:
        print_log("ERROR", f"Error hashing file {filepath}: {e}")
        traceback.print_exc()
        return None
    finally:
        if pbar: pbar.close()


def format_bytes(size_bytes):
    """Formats bytes into a human-readable string (KB, MB, GB)."""
    if not isinstance(size_bytes, (int, float)) or size_bytes < 0:
        return "Invalid size"
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / (1024**2):.2f} MB"
    else:
        return f"{size_bytes / (1024**3):.2f} GB"

def format_duration(seconds):
    """Formats seconds into a human-readable string."""
    if seconds < 0: return "0 seconds"
    if seconds < 60: return f"{seconds:.2f} seconds"
    elif seconds < 3600: return f"{seconds / 60:.2f} minutes"
    else: return f"{seconds / 3600:.2f} hours"
def get_github_credentials(prompt_for_repo=True):
    """Prompts user for GitHub repo (optional) and PAT."""
    github_repo = None
    console = Console() # Instantiate console for prompts

    if prompt_for_repo:
        while not github_repo:
            # Use helper for consistent prompting
            try:
                # --- MODIFIED LINE ---
                repo_input = prompt_user_input(console, f"Enter target GitHub repository {Style.BRIGHT}(owner/repo):{Style.RESET_ALL} ").strip()
                if repo_input and '/' in repo_input and len(repo_input.split('/')) == 2 and all(part.strip() for part in repo_input.split('/')):
                    github_repo = repo_input
                else:
                    print_log("WARN", "Invalid repository format (must be 'owner/repo'). Please try again.")
            except (EOFError, KeyboardInterrupt):
                    print_log("WARN", "\nInput cancelled by user.")
                    return None, None # Indicate cancellation

    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        print_log("INFO", "GITHUB_TOKEN environment variable not set.")
        try:
            # --- NO CHANGE HERE - getpass is needed for hidden input ---
            print_log("INPUT", f"Enter GitHub Personal Access Token (PAT) (input hidden):{Style.RESET_ALL} ")
            github_token = getpass.getpass("").strip()
            if not github_token:
                print_log("WARN", "No token entered.")
                github_token = None
        except (EOFError, KeyboardInterrupt):
            print_log("WARN", "\nToken input cancelled by user.")
            github_token = None # Indicate cancellation
        except Exception as e:
            print_log("ERROR", f"Could not read token securely: {e}.")
            github_token = None
    else:
        print_log("INFO", "Using GitHub token from GITHUB_TOKEN environment variable.")

    return github_repo, github_token

# --- DB Client Functions ---
def db_delete_entry(entry_id: int) -> bool:
    """Requests the server to move an entry to the trash table."""
    print_log("DB", f"Sending request to move entry ID {entry_id} to trash...")
    response = None
    try:
        response = requests.delete(f"{DB_SERVER_URL}/delete_entry/{entry_id}", timeout=15)
        if response.status_code == 404:
             print_log("WARN", f"Server reported entry ID {entry_id} not found for deletion/trashing.")
             return False # Indicate not found rather than general error
        response.raise_for_status() # Raise for other errors (500 etc.)
        print_log("DB", f"Server confirmed entry ID {entry_id} moved to trash.")
        return True
    except (requests.exceptions.RequestException, Exception) as e:
        _handle_db_request_error(f"delete_entry({entry_id})", e, response) # Use helper
        return False
def db_add_log_entry(metadata: Dict[str, Any], github_info: Optional[Dict[str, str]] = None) -> bool:
    """Adds a record of the processed file to the database via the server."""
    print_log("DB", f"Sending log entry request for SHA: {metadata.get('original_sha256', 'N/A')[:8]}...")
    response = None # Initialize response
    try:
        payload = {'metadata': metadata, 'github_info': github_info}
        response = requests.post(f"{DB_SERVER_URL}/add_entry", json=payload, timeout=15)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        response_data = response.json()
        entry_id = response_data.get('id')
        if entry_id:
             print_log("DB", f"Server confirmed log entry added (ID: {entry_id}).")
             return True
        else:
             print_log("WARN", f"Server response OK but no ID returned: {response_data.get('message', 'No message')}")
             return True if response.ok else False # Return True on 2xx

    # --- MODIFIED EXCEPTION HANDLING ---
    except (requests.exceptions.RequestException, json.JSONDecodeError, Exception) as e:
        _handle_db_request_error("add_entry", e, response) # Use helper
        return False
    


import requests # Ensure this is imported at the top of your file
import sys
import time
import traceback # Potentially needed for unexpected errors
from concurrent.futures import ThreadPoolExecutor, as_completed, Future # If you were to parallelize requests (optional)
from typing import Optional, Union, Tuple, Dict, Any, List
# Assuming other necessary imports like Console, Fore, Style, print_log, db_*, _get_*, etc. are present
# Assuming GitHub exception classes like UnknownObjectException, GithubException etc. are imported/defined

# --- Full 'get-download' Main Function (API Redirect Method) ---
# NOTE: This function assumes db_get_entry_details() works correctly.
# The current 500 error originates from the DB server, not this function.

def get_download_main(args: argparse.Namespace):
    """
    Handles listing active entries with GitHub info, prompts user selection,
    uses the GitHub API to resolve, and prints the final asset download URLs.
    """
    mode_display_name = "GET DOWNLOAD URLS (API Method)"
    start_process_time = time.monotonic()
    print_log("STEP", f"Initiating {mode_display_name} using Database Server: {DB_SERVER_URL}")
    console = Console() # Assumes Console class is defined or imported

    # --- Dependency Checks ---
    if not RICH_AVAILABLE: print_log("WARN", "'rich' library not found. Table formatting/prompts basic.")
    if not PYGITHUB_AVAILABLE: print_log("ERROR", "'PyGithub' required for this mode."); sys.exit(1)
    if not REQUESTS_AVAILABLE: print_log("ERROR", "'requests' required for resolving URLs."); sys.exit(1)

    # --- 1. Query Database for Active Entries ---
    available_entries = db_query_entries(query_trash=False) # Assumes db_query_entries exists
    if available_entries is None:
        print_log("ERROR", "Failed to retrieve active entries from database server.")
        sys.exit(1)

    # --- 2. Filter and Display Entries with GitHub Info ---
    print_log("INFO", "Available active entries (with GitHub info):")
    entries_with_gh_info = [
        entry for entry in available_entries
        if entry.get('github_repo') and entry.get('github_release_tag')
    ]
    if not entries_with_gh_info:
        print_log("WARN", "No active entries with associated GitHub repository and release tag information found in the database.")
        sys.exit(0)

    valid_ids = _display_entries_table(entries_with_gh_info, console, is_trash_list=False) # Assumes _display_entries_table exists
    if not valid_ids:
        print_log("INFO", "No valid entries to select from.")
        sys.exit(0)

    # --- 3. Get User Choice ---
    selected_id = None
    while selected_id is None:
        try:
            choice = prompt_user_input(console, "Enter ID to get download URLs for (or 'q' to quit): ").strip() # Assumes prompt_user_input exists
            if choice.lower() == 'q':
                print_log("INFO", "Operation cancelled.")
                sys.exit(0)
            selected_id_int = int(choice)
            if selected_id_int in valid_ids:
                selected_id = selected_id_int
            else:
                print_log("WARN", f"Invalid ID '{selected_id_int}'. Choose an ID from the table that has GitHub info.")
        except ValueError:
            print_log("WARN", "Invalid input. Enter a number or 'q'.")
        except (EOFError, KeyboardInterrupt):
            print_log("WARN", "\nOperation cancelled.")
            sys.exit(130)

    # --- 4. Fetch Full Entry Details ---
    # !!! THIS IS WHERE THE CURRENT 500 ERROR OCCURS !!!
    # !!! The error is in the DB Server, not here. !!!
    print_log("INFO", f"Fetching full details for selected ID: {selected_id}")
    db_details = db_get_entry_details(selected_id) # Assumes db_get_entry_details exists
    # --- The script currently fails here ---
    if not db_details or 'metadata' not in db_details:
        # This error message is shown because db_get_entry_details returns None or invalid data due to the 500 error
        print_log("ERROR", f"Failed retrieving/parsing details for ID {selected_id} from server (check DB server logs for 500 error).")
        sys.exit(1)

    # --- Code below this point will not be reached until the DB server 500 error is fixed ---
    metadata = db_details.get('metadata', {})
    github_repo = db_details.get('github_repo')
    github_tag = db_details.get('github_release_tag')
    original_filename = db_details.get('original_filename', 'N/A')
    num_parts = metadata.get('number_of_parts', -1)
    parts_metadata = metadata.get("parts", [])

    if not github_repo or not github_tag:
        print_log("ERROR", f"Inconsistency: Entry ID {selected_id} selected, but missing GitHub repo/tag in details.")
        sys.exit(1)
    if not isinstance(parts_metadata, list):
        print_log("ERROR", f"Metadata for entry ID {selected_id} is missing or has invalid 'parts' information.")
        sys.exit(1)
    if num_parts < 0:
         print_log("WARN", f"Number of parts is invalid ({num_parts}) in metadata for ID {selected_id}.")

    print_log("INFO", f"Selected: ID={selected_id}, File='{original_filename}', Repo='{github_repo}', Tag='{github_tag}'")

    # --- 5. GitHub Interaction: Connect and Fetch Assets ---
    gh = None; repo = None; release = None; assets = None
    github_token = None # Initialize token variable
    try:
        print_log("STEP", "Connecting to GitHub...")
        _, github_token = get_github_credentials(prompt_for_repo=False) # Assumes get_github_credentials exists
        if not github_token: raise ValueError("GitHub Personal Access Token is required.")

        gh = _get_github_instance(github_token) # Assumes _get_github_instance exists
        if not gh: raise ConnectionError("Failed to authenticate with GitHub.")

        repo = _get_repo(gh, github_repo) # Assumes _get_repo exists
        if not repo: raise ValueError(f"Repository '{github_repo}' not found or access denied.")

        print_log("INFO", f"Fetching release with tag '{github_tag}'...")
        try:
            release = repo.get_release(github_tag)
            print_log("INFO", f"Found release: '{release.title}'")
        except UnknownObjectException: # Assumes UnknownObjectException is defined/imported
            print_log("ERROR", f"Release with tag '{github_tag}' not found in repository '{github_repo}'.")
            sys.exit(1)

        print_log("INFO", "Fetching asset list from release...")
        assets = list(release.get_assets())
        asset_map = {asset.name: asset for asset in assets}
        print_log("INFO", f"Found {len(assets)} assets on the release.")

    except (ValueError, ConnectionError, ConnectionAbortedError, RateLimitExceededException, GithubException) as gh_err:
        print_log("ERROR", f"GitHub interaction failed: {gh_err}")
        if isinstance(gh_err, GithubException): print_log("DEBUG", f"GH Error Data: {gh_err.data}")
        sys.exit(1)
    except Exception as e:
        print_log("ERROR", f"An unexpected error occurred during GitHub interaction: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- 6. Find, Resolve (via API Redirect), and Print Asset URLs ---
    print_log("STEP", f"Resolving and Retrieving Download URLs for '{original_filename}' parts:")
    urls_found = 0
    missing_assets = []
    session = requests.Session() # Use a session

    if num_parts == 0:
        print_log("INFO", "Original file was empty (0 parts). No download URLs to retrieve.")
    elif not parts_metadata and num_parts > 0:
         print_log("WARN", f"Metadata indicates {num_parts} parts, but the 'parts' list is empty. Cannot retrieve URLs.")
    elif parts_metadata:
        try:
             parts_metadata.sort(key=lambda p: p.get('part_number', float('inf')))
        except TypeError as sort_err:
             print_log("WARN", f"Could not sort parts metadata, proceeding in original order. Error: {sort_err}")

        print("-" * 60)
        # Prepare headers for API requests
        api_request_headers = {
            'Authorization': f'token {github_token}',
            'Accept': 'application/octet-stream' # Crucial header to request asset data/redirect
        }

        for part_meta in parts_metadata:
            if not isinstance(part_meta, dict):
                print_log("WARN", f"Skipping invalid item in parts metadata: {part_meta}")
                continue

            part_num = part_meta.get('part_number')
            expected_filename = part_meta.get("part_filename") or (f"part_{part_num}" if part_num is not None else None)
            part_label = f"Part {part_num or 'N/A'} ({expected_filename or 'Unknown Filename'})"

            if not expected_filename:
                print_log("WARN", f"Skipping part with missing number/filename in metadata: {part_meta}")
                continue

            asset = asset_map.get(expected_filename)

            if asset:
                # Use the asset's API URL (e.g., api.github.com/...)
                api_url = getattr(asset, 'url', None)
                if api_url:
                    try:
                        print_log("DEBUG", f"Requesting redirect for {part_label} from API URL: {api_url}", part_num=part_num)
                        # Make HEAD request to the API URL
                        # IMPORTANT: allow_redirects=False to capture the 302 response
                        response = session.head(
                            api_url,
                            headers=api_request_headers,
                            allow_redirects=False, # DO NOT follow redirect automatically
                            timeout=15
                        )

                        # Check if we received the expected redirect status code
                        if response.status_code == 302:
                            final_url = response.headers.get('Location')
                            if final_url:
                                # Successfully extracted the final download URL
                                print(f"  {Fore.CYAN if 'Fore' in globals() else ''}{part_label}:{Style.RESET_ALL if 'Style' in globals() else ''} {Fore.GREEN if 'Fore' in globals() else ''}{final_url}{Style.RESET_ALL if 'Style' in globals() else ''}")
                                urls_found += 1
                            else:
                                # Status was 302, but the Location header was missing (unexpected)
                                print(f"  {Fore.RED if 'Fore' in globals() else ''}{part_label}:{Style.RESET_ALL if 'Style' in globals() else ''} Error: API Redirect (302) received but 'Location' header missing.")
                                missing_assets.append(f"{expected_filename} (Missing Location Header)")
                        # Handle other potential status codes from the API URL
                        elif response.status_code == 200:
                             print(f"  {Fore.YELLOW if 'Fore' in globals() else ''}{part_label}:{Style.RESET_ALL if 'Style' in globals() else ''} Warning: Expected API redirect (302) but received status 200 OK. Asset might be accessible differently?")
                             missing_assets.append(f"{expected_filename} (Unexpected 200 OK)")
                        else:
                            # Any other non-success status from the API asset URL itself
                            error_details = f"Status {response.status_code}"
                            try: # Try to get more details from response body if possible (though HEAD might not have body)
                                 error_details += f": {response.text}"
                            except Exception: pass
                            print(f"  {Fore.RED if 'Fore' in globals() else ''}{part_label}:{Style.RESET_ALL if 'Style' in globals() else ''} Error: Unexpected status from API URL: {error_details}")
                            missing_assets.append(f"{expected_filename} (API Status {response.status_code})")

                    except requests.exceptions.Timeout:
                         print(f"  {Fore.RED if 'Fore' in globals() else ''}{part_label}:{Style.RESET_ALL if 'Style' in globals() else ''} Timeout contacting GitHub API URL.")
                         missing_assets.append(f"{expected_filename} (API Timeout)")
                    except requests.exceptions.RequestException as req_err:
                        # Includes ConnectionError, etc.
                        print(f"  {Fore.RED if 'Fore' in globals() else ''}{part_label}:{Style.RESET_ALL if 'Style' in globals() else ''} Network Error contacting GitHub API URL: {req_err}")
                        missing_assets.append(f"{expected_filename} (API Network Error)")
                    except Exception as e:
                         print(f"  {Fore.RED if 'Fore' in globals() else ''}{part_label}:{Style.RESET_ALL if 'Style' in globals() else ''} Unexpected error resolving URL via API: {e}")
                         missing_assets.append(f"{expected_filename} (API Resolve Error)")

                else: # Asset object existed but had no 'url' attribute
                    print(f"  {Fore.YELLOW if 'Fore' in globals() else ''}{part_label}:{Style.RESET_ALL if 'Style' in globals() else ''} Asset found but API URL ('url' attribute) missing.")
                    missing_assets.append(f"{expected_filename} (Missing API URL)")
            else: # Asset not found on release matching expected_filename
                print(f"  {Fore.RED if 'Fore' in globals() else ''}{part_label}:{Style.RESET_ALL if 'Style' in globals() else ''} Asset NOT FOUND on GitHub release.")
                missing_assets.append(f"{expected_filename} (Asset missing)")

        print("-" * 60)

    # --- 7. Final Summary ---
    end_process_time = time.monotonic()
    total_time = end_process_time - start_process_time

    print_log("INFO", f"URL Retrieval Summary for Entry ID {selected_id}:")
    expected_parts = num_parts if num_parts >= 0 else len(parts_metadata)
    print_log("INFO", f"  Expected Parts: {expected_parts}")
    print_log("INFO", f"  URLs Found:     {urls_found}")
    if missing_assets:
        max_missing_display = 5
        display_missing = missing_assets[:max_missing_display]
        if len(missing_assets) > max_missing_display:
             display_missing.append(f"...and {len(missing_assets) - max_missing_display} more")
        print_log("WARN", f"  Missing/Failed: {len(missing_assets)} ({', '.join(display_missing)})")

    if urls_found == expected_parts and expected_parts >= 0 and not missing_assets:
        print_log("SUCCESS", f"{mode_display_name} complete. All expected URLs retrieved.")
        exit_code = 0
    elif urls_found > 0:
         print_log("WARN", f"{mode_display_name} complete, but some assets/URLs were missing or failed.")
         exit_code = 1
    elif expected_parts == 0:
         print_log("SUCCESS", f"{mode_display_name} complete (original file was empty).")
         exit_code = 0
    else:
         print_log("ERROR", f"{mode_display_name} failed. No valid URLs retrieved.")
         exit_code = 1

    print_log("INFO", f"Total time: {format_duration(total_time)}") # Assumes format_duration exists
    sys.exit(exit_code)

# --- End of get_download_main function ---






# Modify the signature
def db_query_entries(query_trash: bool = False) -> Optional[List[Dict[str, Any]]]:
    """Queries the database for active or trashed entries via the server."""
    # Add logic to change endpoint and log message
    endpoint = "/entries/trash" if query_trash else "/entries"
    log_msg = "trashed" if query_trash else "active"
    print_log("DB", f"Querying {log_msg} entries from DB server: {DB_SERVER_URL}{endpoint}")
    response = None # Add this line
    try:
        # Use the dynamic endpoint
        response = requests.get(f"{DB_SERVER_URL}{endpoint}", timeout=15)
        response.raise_for_status()
        entries = response.json()
        # Use dynamic log message
        print_log("DB", f"Received {len(entries)} {log_msg} entries from server.")
        return entries
    # Update exception handling (ideally call _handle_db_request_error)
    except (requests.exceptions.RequestException, json.JSONDecodeError, Exception) as e:
         _handle_db_request_error(f"query_{log_msg}_entries", e, response) # Assumes helper
         # Or handle individually
         print_log("ERROR", f"DB Client Error (query_{log_msg}_entries): Failed: {e}")
         return None
def db_get_entry_details(entry_id: int) -> Optional[Dict[str, Any]]:
    """Fetches full details for a specific *active* entry ID via the server."""
    print_log("DB", f"Fetching details for active entry ID {entry_id} from server...")
    response = None # Initialize response
    try:
        response = requests.get(f"{DB_SERVER_URL}/entry/{entry_id}", timeout=15)
        if response.status_code == 404:
            print_log("WARN", f"Server reported active entry ID {entry_id} not found.")
            return None
        response.raise_for_status() # Raise for other errors (500 etc.)

        details = response.json()
        if details and 'metadata' in details: # Basic check
            print_log("DB", f"Received details for active entry ID {entry_id}.")
            return details
        else:
            print_log("ERROR", f"DB Client Error (get_details): Invalid/incomplete data for ID {entry_id}: {details}")
            return None

    # --- MODIFIED EXCEPTION HANDLING ---
    except (requests.exceptions.RequestException, json.JSONDecodeError, Exception) as e:
        _handle_db_request_error(f"get_entry_details({entry_id})", e, response) # Use helper
        return None

def _handle_db_request_error(operation: str, error: Exception, response: Optional[requests.Response] = None):
    """Centralized handler for DB request errors."""
    if isinstance(error, requests.exceptions.ConnectionError):
        print_log("ERROR", f"DB Client Error ({operation}): Could not connect to server at {DB_SERVER_URL}. Is it running?")
    elif isinstance(error, requests.exceptions.Timeout):
        print_log("ERROR", f"DB Client Error ({operation}): Request timed out to {DB_SERVER_URL}.")
    elif isinstance(error, requests.exceptions.HTTPError):
        status = response.status_code if response is not None else "N/A"
        resp_text = response.text if response is not None else "No response text."
        try: # Try to get JSON error message from server
             err_detail = response.json().get('error', resp_text) if response is not None else resp_text
        except json.JSONDecodeError:
             err_detail = resp_text
        print_log("ERROR", f"DB Client Error ({operation}): Server returned status {status}. Message: {err_detail}")
    elif isinstance(error, json.JSONDecodeError):
        resp_text = response.text if response is not None else "No response text."
        print_log("ERROR", f"DB Client Error ({operation}): Could not decode server JSON response: {resp_text[:200]}...") # Limit length
    elif isinstance(error, requests.exceptions.RequestException):
        print_log("ERROR", f"DB Client Error ({operation}): General request failed: {error}")
    else:
        print_log("ERROR", f"DB Client Error ({operation}): Unexpected error: {error}")
        traceback.print_exc()

# --- GitHub API Integration Functions ---
# <<< INSERTED FULL IMPLEMENTATIONS >>>

def _get_github_instance(token: str) -> Github:
    """Initializes and returns an authenticated PyGithub instance."""
    if not PYGITHUB_AVAILABLE:
        raise ImportError("PyGithub library required. Install: pip install PyGithub")
    if not token:
        raise ValueError("GitHub token is required.")
    try:
        gh = Github(token, timeout=60)
        user = gh.get_user()
        login = user.login
        print_log("INFO", f"Successfully authenticated with GitHub as {login}.", clear_line=True) # Clear potential prompt line
        try:
            # Log rate limit info
            rate_limit = gh.get_rate_limit().core
            reset_time_str = rate_limit.reset.strftime('%Y-%m-%d %H:%M:%S')
            print_log("DEBUG", f"GitHub Rate Limit: {rate_limit.remaining}/{rate_limit.limit}, Resets: {reset_time_str}")
            if rate_limit.remaining < 20:
                 print_log("WARN", f"Low GitHub API rate limit remaining ({rate_limit.remaining}).")
        except GithubException as rl_e:
             print_log("WARN", f"Could not retrieve rate limit info: {rl_e.data.get('message', str(rl_e))}")
        return gh
    except (RateLimitExceededException, BadCredentialsException, TwoFactorException) as auth_e:
        error_msg = f"GitHub API Error: {auth_e.data.get('message', str(auth_e))}"
        if isinstance(auth_e, RateLimitExceededException):
            reset_time_ts = auth_e.headers.get('x-ratelimit-reset', time.time())
            reset_time = datetime.datetime.fromtimestamp(float(reset_time_ts))
            wait_seconds = max(0, (reset_time - datetime.datetime.now()).total_seconds()) + 5
            error_msg = f"GitHub Rate Limit Exceeded. Try again after {reset_time.strftime('%Y-%m-%d %H:%M:%S')} (approx {wait_seconds/60:.1f} mins)."
            raise ConnectionAbortedError(error_msg) from auth_e
        elif isinstance(auth_e, BadCredentialsException):
            error_msg = "GitHub Authentication Failed: Bad credentials (Invalid Token?)." # More specific
        elif isinstance(auth_e, TwoFactorException):
            error_msg = "GitHub Authentication Failed: Two-factor auth required."
        print_log("ERROR", error_msg, clear_line=True) # Clear potential prompt line
        raise ConnectionError(error_msg) from auth_e
    except GithubException as e:
        raise ConnectionError(f"GitHub auth failed (Status: {e.status}): {e.data.get('message', 'Unknown')}") from e
    except Exception as e:
        raise ConnectionError(f"Unexpected error during GitHub authentication: {e}") from e

def _get_repo(gh: Github, repo_full_name: str) -> Optional[Repository]:
    """Gets a Repository object, handling errors."""
    try:
        repo = gh.get_repo(repo_full_name)
        print_log("DEBUG", f"Found repository: {repo.full_name}")
        return repo
    except UnknownObjectException:
        print_log("WARN", f"Repository '{repo_full_name}' not found or access denied.")
        return None
    except RateLimitExceededException as e:
        reset_time_ts = e.headers.get('x-ratelimit-reset', time.time())
        reset_time = datetime.datetime.fromtimestamp(float(reset_time_ts))
        error_msg = f"GitHub Rate Limit Exceeded accessing repo. Try again after {reset_time.strftime('%Y-%m-%d %H:%M:%S')}."
        print_log("ERROR", error_msg)
        raise ConnectionAbortedError(error_msg) from e
    except GithubException as e:
        print_log("ERROR", f"Error accessing repository '{repo_full_name}' (Status: {e.status}): {e.data.get('message', 'Unknown error')}")
        raise
    except Exception as e:
        print_log("ERROR", f"Unexpected error getting repository '{repo_full_name}': {e}")
        raise

def _check_release_exists_internal(repo: Repository, tag_name: str) -> Tuple[bool, Optional[GitRelease]]:
    """Checks if a release with the given tag exists, handling errors."""
    print_log("INFO", f"Checking for release with tag '{tag_name}'...")
    try:
        release = repo.get_release(tag_name)
        print_log("INFO", f"Found existing release with tag '{tag_name}'.")
        return True, release
    except UnknownObjectException:
        print_log("INFO", f"No existing release found with tag '{tag_name}'.")
        return False, None
    except RateLimitExceededException as e:
        reset_time_ts = e.headers.get('x-ratelimit-reset', time.time())
        reset_time = datetime.datetime.fromtimestamp(float(reset_time_ts))
        error_msg = f"GitHub Rate Limit Exceeded checking release. Try again after {reset_time.strftime('%Y-%m-%d %H:%M:%S')}."
        print_log("ERROR", error_msg)
        raise ConnectionAbortedError(error_msg) from e
    except GithubException as e:
        print_log("ERROR", f"Error checking for release tag '{tag_name}' (Status: {e.status}): {e.data.get('message', 'Unknown error')}")
        raise
    except Exception as e:
        print_log("ERROR", f"Unexpected error checking release tag '{tag_name}': {e}")
        raise

def _ensure_repository_exists(gh: Github, repo_full_name: str, private: bool, description: str, readme_content: Optional[str]) -> Repository:
    """Gets repo if exists, creates if not. Handles errors."""
    print_log("INFO", f"Ensuring repository '{repo_full_name}' exists...")
    owner, repo_name = repo_full_name.split('/')
    try:
        repo = gh.get_repo(repo_full_name)
        print_log("INFO", f"Repository '{repo.full_name}' already exists.")
        return repo
    except UnknownObjectException:
        print_log("INFO", f"Repository '{repo_full_name}' not found. Attempting creation...")
        try:
            user = gh.get_user()
            target_entity = user
            if owner.lower() != user.login.lower():
                 try:
                     org = gh.get_organization(owner)
                     target_entity = org
                     print_log("DEBUG", f"Targeting organization '{owner}' for repo creation.")
                 except UnknownObjectException:
                     raise ValueError(f"Organization '{owner}' not found or user '{user.login}' cannot create repositories there.") from None

            repo = target_entity.create_repo(
                 name=repo_name, private=private, description=description, auto_init=False
            )
            print_log("SUCCESS", f"Repository '{repo.full_name}' created successfully.")

            if readme_content:
                 try:
                     time.sleep(3) # Increased wait time
                     default_branch_name = repo.default_branch
                     repo.create_file(
                         path="README.md", message="Initial commit: Add README.md",
                         content=readme_content, branch=default_branch_name
                     )
                     print_log("INFO", f"Added README.md to '{repo.full_name}'.")
                 except GithubException as readme_e:
                     print_log("WARN", f"Failed to add README.md (Status: {readme_e.status}): {readme_e.data.get('message', 'Unknown error')}")
            return repo
        except RateLimitExceededException as e: raise
        except GithubException as create_e:
            err_data = create_e.data or {}
            err_msg = str(err_data.get('message','')).lower()
            err_errors = err_data.get('errors', [])
            name_exists = "name already exists" in err_msg or \
                           any("already exists" in str(e.get('message','')).lower() for e in err_errors if isinstance(e, dict))

            if create_e.status == 422 and name_exists:
                print_log("WARN", f"Repository '{repo_full_name}' existed despite initial check (race?). Attempting retrieval...")
                try:
                    time.sleep(2)
                    return gh.get_repo(repo_full_name)
                except RateLimitExceededException as get_rl: raise get_rl
                except Exception as get_e: raise RuntimeError(f"Failed to get repo '{repo_full_name}' after creation conflict: {get_e}") from get_e
            else:
                raise RuntimeError(f"Failed to create repository '{repo_full_name}' (Status: {create_e.status}): {err_data.get('message', 'Unknown error')}") from create_e
        except Exception as e:
            raise RuntimeError(f"Unexpected error ensuring repository '{repo_full_name}' exists: {e}") from e

    except RateLimitExceededException as e: raise
    except GithubException as e:
        raise RuntimeError(f"Error accessing repository '{repo_full_name}' (Status: {e.status}): {e.data.get('message', 'Unknown error')}") from e
    except Exception as e:
         raise RuntimeError(f"Unexpected error accessing repository '{repo_full_name}': {e}") from e

def _create_release_internal(repo: Repository, tag_name: str, release_name: str, body: str) -> GitRelease:
    """Creates a new release, handling errors."""
    print_log("INFO", f"Creating release '{release_name}' with tag '{tag_name}'...")
    try:
        release = repo.create_git_release(
            tag=tag_name, name=release_name, message=body, draft=False, prerelease=False
        )
        print_log("SUCCESS", f"Release '{release.tag_name}' created successfully. URL: {release.html_url}")
        return release
    except RateLimitExceededException as e: raise
    except GithubException as e:
        err_data = e.data or {}
        err_msg = str(err_data.get('message','')).lower()
        err_errors = err_data.get('errors', [])
        already_exists = "already exists" in err_msg or \
                          any("already_exists" in str(err.get('resource','')).lower() and "tag" in str(err.get('field','')).lower() for err in err_errors if isinstance(err, dict))

        if e.status == 422 and already_exists:
            print_log("WARN", f"Release with tag '{tag_name}' already exists (race?). Attempting retrieval.")
            try:
                time.sleep(1)
                return repo.get_release(tag_name)
            except RateLimitExceededException as get_rl: raise get_rl
            except Exception as get_e:
                raise RuntimeError(f"Failed to get existing release '{tag_name}' after conflict: {get_e}") from get_e
        else:
            raise RuntimeError(f"Failed to create release '{tag_name}' (Status: {e.status}): {err_data.get('message', 'Unknown error')}") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error creating release '{tag_name}': {e}") from e

def _upload_asset_internal(release: GitRelease, asset_path: str, asset_name: str) -> Tuple[bool, bool, Optional[GitReleaseAsset]]:
    """Uploads a single asset to a release, using animation."""
    if not os.path.isfile(asset_path):
        raise FileNotFoundError(f"Asset file not found for upload: {asset_path}")

    part_num_str = asset_name.split('_')[-1] if asset_name.startswith("part_") else None
    upload_success = False; skipped = False; asset_info = None
    stop_event = threading.Event()
    animation_thread = threading.Thread(target=upload_animation, args=(stop_event, asset_name), daemon=True)

    try:
        animation_thread.start()
        # Blocking call to upload the asset
        uploaded_asset = release.upload_asset(path=asset_path, name=asset_name)
        asset_info = uploaded_asset # Store info before stopping animation
        upload_success = True
    except RateLimitExceededException as e: raise # Propagate specific errors
    except GithubException as e:
        err_data = e.data or {}
        err_errors = err_data.get('errors', [])
        already_exists = any(err.get('code') == 'already_exists' for err in err_errors if isinstance(err, dict))

        if e.status == 422 and already_exists:
            skipped = True
            try: # Try to get existing asset info
                asset_info = next((a for a in release.get_assets() if a.name == asset_name), None)
            except Exception: pass # Ignore errors getting existing asset
        else:
            # Log actual upload errors here
            print_log("ERROR", f"Failed to upload asset '{asset_name}' (Status: {e.status}): {err_data.get('message', 'Unknown error')}", part_num=part_num_str)
            raise # Re-raise the exception
    except Exception as e:
        # Log unexpected errors here
        print_log("ERROR", f"Unexpected error uploading asset '{asset_name}': {e}", part_num=part_num_str)
        raise # Re-raise the exception
    finally:
        # Stop Animation and Print Final Status
        stop_event.set()
        animation_thread.join(timeout=1) # Wait briefly

        if upload_success:
            size_str = format_bytes(asset_info.size) if asset_info else "Unknown size"
            print_log("SUCCESS", f"Uploaded asset '{asset_name}' ({size_str}).", part_num=part_num_str, clear_line=True)
        elif skipped:
            print_log("WARN", f"Asset '{asset_name}' already exists on release '{release.tag_name}'. Skipped.", part_num=part_num_str, clear_line=True)
            if asset_info: print_log("DEBUG", f"Retrieved existing asset info for '{asset_name}'.")
            else: print_log("WARN", "Could not retrieve existing asset info.")
        else:
            # Error logging happened in the except blocks, ensure line is clear
            print_log("ERROR", "", part_num=part_num_str, clear_line=True)

    return upload_success, skipped, asset_info





# --- Helper Functions ---


def prompt_user_input(console: Console, prompt_message: str) -> str:
    """Handles user input using Rich Console if available, placing the prompt below the log prefix and a separator."""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    thread_name = threading.current_thread().name
  
    # Define the indicator for the actual input line
    indicator = f"{Fore.YELLOW}--> {Style.RESET_ALL}"

    # Use Rich's input, combining the indicator and the core message
    # This will appear below the separator
    try:
        user_response = console.input(f"{indicator}{prompt_message}")
    except (EOFError, KeyboardInterrupt) as e:
        # Ensure the line is cleared if input is cancelled mid-prompt
        print() # Move to a new line after cancellation
        raise e # Re-raise the exception

    return user_response

def prompt_user_confirm(console: Console, prompt_message: str, default_val: bool = False) -> bool:
    """Handles yes/no confirmation using Rich Confirm if available, placing the prompt below the log prefix and a separator."""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    thread_name = threading.current_thread().name
  
    # Define the indicator for the actual input line
    indicator = f"{Fore.YELLOW}--> {Style.RESET_ALL}"

    # Use Rich's Confirm.ask, combining the indicator and the core message.
    # Rich will append the [y/n] options automatically.
    # This will appear below the separator
    try:
        confirmed = Confirm.ask(prompt=f"{indicator}{prompt_message}", default=default_val, console=console)
    except (EOFError, KeyboardInterrupt) as e:
        # Ensure the line is cleared if input is cancelled mid-prompt
        print() # Move to a new line after cancellation
        raise e # Re-raise the exception
    return confirmed



def _delete_github_asset(asset: GitReleaseAsset, asset_name: str, part_num_str: Optional[str]) -> bool:
    """Deletes a single asset from GitHub, handling errors. Returns True on success/not found, False on failure."""
    print_log("INFO", f"Attempting to delete GitHub asset: '{asset_name}'...", part_num=part_num_str)
    if not asset: # Should not happen if called correctly, but safety check
        print_log("WARN", f"Asset object for '{asset_name}' is None. Cannot delete.", part_num=part_num_str)
        return False
    try:
        if asset.delete_asset():
             print_log("SUCCESS", f"Deleted GitHub asset '{asset_name}'.", part_num=part_num_str)
             return True
        else:
             # PyGithub's delete_asset returns True on success (204 No Content), False otherwise.
             # It might not raise an exception for non-204 responses sometimes.
             print_log("WARN", f"GitHub asset '{asset_name}' deletion call returned False (API might have failed silently).", part_num=part_num_str)
             return False
    except RateLimitExceededException as e:
        reset_time_ts = e.headers.get('x-ratelimit-reset', time.time())
        reset_time = datetime.datetime.fromtimestamp(float(reset_time_ts))
        print_log("ERROR", f"Rate limit exceeded deleting asset '{asset_name}'. Try again after {reset_time.strftime('%Y-%m-%d %H:%M:%S')}.", part_num=part_num_str)
        return False # Indicate failure due to rate limit for this asset
    except UnknownObjectException:
        # If the asset we tried to delete was already gone
        print_log("WARN", f"GitHub asset '{asset_name}' not found for deletion (already deleted?).", part_num=part_num_str)
        return True # Consider it success if it's already gone
    except GithubException as e:
        # Handle other specific GitHub errors during deletion
        print_log("ERROR", f"Failed to delete asset '{asset_name}' (Status: {e.status}): {e.data.get('message', 'Unknown error')}", part_num=part_num_str)
        return False
    except Exception as e:
        # Catch any other unexpected errors
        print_log("ERROR", f"Unexpected error deleting asset '{asset_name}': {e}", part_num=part_num_str)
        traceback.print_exc()
        return False



# --- Core Part Processing Functions ---
def process_plaintext_part(part_info: Dict[str, Any]) -> Dict[str, Any]:
    """Reads, writes plaintext part, returns metadata."""
    part_num = part_info['part_number']; source_path = part_info['source_path']
    offset = part_info['offset']; size = part_info['size']
    output_dir = part_info['output_dir']; hash_buffer_size = part_info['hash_buffer_size']

    print_log("INFO", f"Starting processing (plaintext).", part_num=part_num)
    part_start_time = time.monotonic()
    part_filepath = None

    try:
        print_log("DEBUG", f"Reading {format_bytes(size)} from offset {offset}...", part_num=part_num)
        with open(source_path, 'rb') as infile:
            infile.seek(offset)
            part_plaintext_data = infile.read(size)
        if len(part_plaintext_data) != size: raise IOError(f"Read size mismatch for part {part_num}.")

        part_filename = f"part_{part_num}" # <-- Filename defined here
        part_filepath = os.path.join(output_dir, part_filename)
        print_log("DEBUG", f"Writing plaintext data to '{part_filepath}'...", part_num=part_num)
        with open(part_filepath, 'wb') as outfile:
            outfile.write(part_plaintext_data)
        plaintext_file_size = len(part_plaintext_data)
        del part_plaintext_data

        print_log("DEBUG", f"Calculating SHA-256 of plaintext part file...", part_num=part_num)
        plaintext_part_sha256 = calculate_sha256(part_filepath, plaintext_file_size, hash_buffer_size, desc_prefix=f"Part {part_num} Hash")
        if not plaintext_part_sha256: raise Exception(f"SHA256 failed for plaintext part {part_num}")

        part_end_time = time.monotonic()
        print_log("SUCCESS", f"Finished processing (plaintext). Time: {format_duration(part_end_time - part_start_time)}.", part_num=part_num)
        # --- MODIFIED RETURN VALUE ---
        return {"part_number": part_num, "plaintext_sha256": plaintext_part_sha256, "part_filename": part_filename}
    except Exception as e:
        print_log("ERROR", f"FAILED (plaintext)! Error: {e}", part_num=part_num)
        if part_filepath and os.path.exists(part_filepath):
            try: os.remove(part_filepath); print_log("WARN", f"Cleaned up partial file '{part_filepath}'", part_num=part_num)
            except Exception as rm_e: print_log("WARN", f"Cleanup failed for '{part_filepath}': {rm_e}", part_num=part_num)
        raise Exception(f"Part {part_num} failed plaintext processing.") from e
def process_encryption_part(part_info: Dict[str, Any]) -> Dict[str, Any]:
    """Reads, encrypts, writes encrypted part, returns metadata."""
    if not CRYPTOGRAPHY_AVAILABLE: raise RuntimeError("Cryptography library missing for encryption.")
    part_num = part_info['part_number']; source_path = part_info['source_path']
    offset = part_info['offset']; size = part_info['size']
    output_dir = part_info['output_dir']; hash_buffer_size = part_info['hash_buffer_size']

    print_log("INFO", f"Starting processing (encryption).", part_num=part_num)
    part_start_time = time.monotonic()
    part_filepath = None

    try:
        print_log("DEBUG", f"Reading {format_bytes(size)} from offset {offset}...", part_num=part_num)
        with open(source_path, 'rb') as infile:
            infile.seek(offset)
            part_plaintext_data = infile.read(size)
        if len(part_plaintext_data) != size: raise IOError(f"Read size mismatch for part {part_num}.")

        print_log("DEBUG", f"Encrypting {format_bytes(len(part_plaintext_data))}...", part_num=part_num)
        aes_key = AESGCM.generate_key(bit_length=256)
        aes_nonce = os.urandom(AES_GCM_NONCE_SIZE)
        aesgcm = AESGCM(aes_key)
        encrypted_data = aesgcm.encrypt(aes_nonce, part_plaintext_data, None)
        del part_plaintext_data

        part_filename = f"part_{part_num}" # <-- Filename defined here
        part_filepath = os.path.join(output_dir, part_filename)
        print_log("DEBUG", f"Writing encrypted data to '{part_filepath}'...", part_num=part_num)
        with open(part_filepath, 'wb') as outfile:
            outfile.write(aes_nonce)
            outfile.write(encrypted_data)
        encrypted_file_size = len(aes_nonce) + len(encrypted_data)
        del encrypted_data

        print_log("DEBUG", f"Calculating SHA-256 of encrypted file...", part_num=part_num)
        encrypted_part_sha256 = calculate_sha256(part_filepath, encrypted_file_size, hash_buffer_size, desc_prefix=f"Enc Part {part_num} Hash")
        if not encrypted_part_sha256: raise Exception(f"SHA256 failed for encrypted part {part_num}")

        part_end_time = time.monotonic()
        print_log("SUCCESS", f"Finished processing (encryption). Time: {format_duration(part_end_time - part_start_time)}.", part_num=part_num)
        # --- MODIFIED RETURN VALUE ---
        return {
            "part_number": part_num,
            "encrypted_sha256": encrypted_part_sha256,
            "decryption_key_base64": base64.b64encode(aes_key).decode('utf-8'),
            "part_filename": part_filename # Added filename
        }
    except Exception as e:
        print_log("ERROR", f"FAILED (encryption)! Error: {e}", part_num=part_num)
        if part_filepath and os.path.exists(part_filepath):
            try: os.remove(part_filepath); print_log("WARN", f"Cleaned up partial file '{part_filepath}'", part_num=part_num)
            except Exception as rm_e: print_log("WARN", f"Cleanup failed for '{part_filepath}': {rm_e}", part_num=part_num)
        raise Exception(f"Part {part_num} failed encryption processing.") from e
def process_decryption_part(part_info: Dict[str, Any]) -> Tuple[int, Optional[bytes], bool]:
    """Verifies, reads, decrypts encrypted part, returns plaintext and success flag."""
    if not CRYPTOGRAPHY_AVAILABLE: raise RuntimeError("Cryptography library missing for decryption.")
    part_num = part_info['part_number']; part_filepath = part_info['part_filepath']
    expected_sha256 = part_info['expected_sha256'] # SHA of the *encrypted* file
    decryption_key = part_info['decryption_key']
    hash_buffer_size = part_info['hash_buffer_size']
    layer_status = part_info.get('layer_status')
    status_lock = part_info.get('status_lock')

    def _update_status(new_status):
        if layer_status is not None and status_lock:
            with status_lock:
                layer_status[part_filename] = new_status

    part_filename = f"part_{part_num}"
    print_log("INFO", f"Starting processing (decryption).", part_num=part_num)
    _update_status(LAYER_WAITING_EXTRACT) # Mark as queued
    part_start_time = time.monotonic()

    try:
        print_log("DEBUG", f"Verifying integrity of '{os.path.basename(part_filepath)}' before decryption...", part_num=part_num)
        if not os.path.isfile(part_filepath): raise FileNotFoundError(f"Part file not found: {part_filepath}")
        part_file_size = os.path.getsize(part_filepath)
        if part_file_size <= AES_GCM_NONCE_SIZE: raise ValueError(f"Part file too small/corrupted: {part_filepath}")

        actual_sha256 = calculate_sha256(part_filepath, part_file_size, hash_buffer_size, desc_prefix=f"Verify Encrypted {part_num}")
        if not actual_sha256: raise Exception(f"Hash calculation failed for encrypted part {part_num}")
        if actual_sha256 != expected_sha256: raise ValueError(f"Integrity FAIL (Encrypted Data) part {part_num}. Exp {expected_sha256[:8]}.., Got {actual_sha256[:8]}..")
        print_log("DEBUG", f"Integrity check passed (encrypted data).", part_num=part_num)

        print_log("INFO", f"Starting extraction...", part_num=part_num) # Changed log message
        _update_status(LAYER_EXTRACTING)

        print_log("DEBUG", f"Reading nonce and encrypted data...", part_num=part_num)
        with open(part_filepath, 'rb') as infile:
            nonce = infile.read(AES_GCM_NONCE_SIZE)
            encrypted_data_with_tag = infile.read()
        if len(nonce) != AES_GCM_NONCE_SIZE: raise IOError("Could not read full nonce.")
        if not encrypted_data_with_tag: raise IOError("Could not read encrypted data + tag.")

        print_log("DEBUG", f"Decrypting {format_bytes(len(encrypted_data_with_tag))}...", part_num=part_num)
        aesgcm = AESGCM(decryption_key)
        decrypted_data = aesgcm.decrypt(nonce, encrypted_data_with_tag, None)

        part_end_time = time.monotonic()
        _update_status(LAYER_EXTRACTED)
        print_log("SUCCESS", f"Layer extracted. Time: {format_duration(part_end_time - part_start_time)}.", part_num=part_num)
        return (part_num, decrypted_data, True) # Return success flag

    except InvalidTag:
        _update_status(LAYER_EXTRACT_FAILED)
        print_log("ERROR", f"Extraction FAILED: Auth tag mismatch! Layer {part_num} corrupt/wrong key.", part_num=part_num)
    except Exception as e:
        _update_status(LAYER_EXTRACT_FAILED)
        print_log("ERROR", f"Extraction FAILED! Error: {e}", part_num=part_num)

    return (part_num, None, False) # Return failure


def upload_to_github_release(gh: Github, output_dir: str, metadata: dict, repo_full_name: str, upload_json_file: bool = False) -> Optional[Dict[str, str]]:
    """
    Handles GitHub release creation/upload, sanitizing the tag.
    Returns repo, original tag, and sanitized tag info on success.
    """
    active_mode = metadata.get("mode", "unknown")
    print_log("STEP", f"Starting GitHub Release Upload ({active_mode} mode) for repo: {repo_full_name}")
    json_action = "including" if upload_json_file else "excluding"
    print_log("INFO", f"Initiating upload ({json_action} metadata JSON).")

    github_info = None # Will store repo_name, original_tag, sanitized_tag

    try:
        # 1. Ensure Repository Exists (Logic remains the same)
        repo_short_name = repo_full_name.split('/')[-1]
        repo_readme = f"# {repo_short_name}\n\nFile parts storage ({active_mode} mode).\n" \
                      f"Original SHA256: {metadata.get('original_sha256', 'N/A')}\n" \
                      f"Managed by file processor v{PROCESSOR_VERSION}.\n" \
                      f"Created: {datetime.datetime.now(datetime.timezone.utc).isoformat()}"

        repo = _ensure_repository_exists(gh, repo_full_name, private=True, description=f"File Processor Storage ({active_mode})", readme_content=repo_readme)

        # 2. Prepare Release Details & Sanitize Tag
        original_filename = metadata.get("original_filename", "unknown_file")
        original_sha256 = metadata.get("original_sha256", "unknown_sha")
        original_sha256_short = original_sha256[:8]
        num_parts = metadata.get("number_of_parts", 0)
        creation_date = metadata.get("creation_datetime_utc", "N/A")
        processing_time_str = metadata.get("total_processing_time", "N/A")

        # --- MODIFICATION START ---
        # Construct the *proposed* tag first
        proposed_release_tag = f"file-{active_mode}-{original_filename}-{original_sha256_short}"
        
        # Sanitize the proposed tag
        sanitized_release_tag = sanitize_release_version_tag(proposed_release_tag)
        
        release_name = f"{original_filename} ({original_sha256_short})" # Release name usually has fewer restrictions
        
        print_log("INFO", f"Proposed tag: '{proposed_release_tag}'")
        print_log("INFO", f"Sanitized tag for release: '{sanitized_release_tag}'")
        # --- MODIFICATION END ---

        json_upload_notice = "**Metadata JSON uploaded as asset.**" if upload_json_file else \
                             f"**Metadata JSON NOT uploaded (contains keys for mode '{active_mode}'). Stored in DB.**"
        release_body = f"""\
**File Parts ({active_mode.capitalize()} Mode)**
* **Original Filename:** `{original_filename}`
* **Original SHA256:** `{original_sha256}`
* **Mode:** `{active_mode}`
* **Number of Parts:** {num_parts}
* **Processed On (UTC):** {creation_date}
* **Processing Time:** {processing_time_str}

**Release Tag Info:**
* **Original Proposed Tag:** `{proposed_release_tag}`
* **Sanitized Tag Used:** `{sanitized_release_tag}`

**IMPORTANT:** {json_upload_notice}
(Processor Version: {PROCESSOR_VERSION})
"""
        print_log("INFO", f"Preparing release: Name='{release_name}', Sanitized Tag='{sanitized_release_tag}'")

        # 3. Create GitHub Release using the *sanitized* tag
        # --- MODIFICATION START ---
        release = _create_release_internal(repo, sanitized_release_tag, release_name, release_body)
        # --- MODIFICATION END ---
        
        release_url = release.html_url
        
        # --- MODIFICATION START ---
        # Store repo name, original proposed tag, and the final sanitized tag used
        github_info = {
            "repo_full_name": repo.full_name, 
            "original_release_tag": proposed_release_tag, 
            "sanitized_release_tag": release.tag_name # Use the tag confirmed by GH API
        }
        # --- MODIFICATION END ---

        # 4. Prepare List of Assets (Logic remains the same)
        print_log("INFO", "Gathering assets for upload...")
        files_to_upload = []
        parts_list = metadata.get("parts", [])
        if not isinstance(parts_list, list): raise ValueError("Metadata 'parts' key is not a list.")

        expected_part_filenames = [f"part_{p['part_number']}" for p in parts_list if isinstance(p, dict) and 'part_number' in p]
        if len(expected_part_filenames) != num_parts:
            print_log("WARN", f"Metadata part count mismatch: Expected {num_parts}, found {len(expected_part_filenames)}.")

        for part_filename in expected_part_filenames:
            item_path = os.path.join(output_dir, part_filename)
            if os.path.isfile(item_path): files_to_upload.append(item_path)
            else: raise FileNotFoundError(f"Expected part file '{part_filename}' not found in '{output_dir}'.")

        if upload_json_file:
            json_filename = f"{original_sha256}.json"
            json_file_path_to_upload = os.path.join(output_dir, json_filename)
            if not os.path.isfile(json_file_path_to_upload):
                raise FileNotFoundError(f"JSON file '{json_filename}' required for upload but not found.")
            files_to_upload.append(json_file_path_to_upload)

        if not files_to_upload and num_parts > 0:
            raise ValueError(f"No files found to upload, but expected {num_parts} parts.")
        elif not files_to_upload and num_parts == 0:
            print_log("INFO", "Original file empty. No assets to upload.")
            print_log("SUCCESS", f"Upload complete (no assets needed). Release URL: {release_url}")
            return github_info # Return collected info even if no assets uploaded

        # 5. Upload Assets Loop (Logic remains the same)
        print_log("STEP", f"Uploading {len(files_to_upload)} asset(s) to release '{release.tag_name}'...")
        upload_successful_count = 0; upload_skipped_count = 0; upload_failed = False
        total_files = len(files_to_upload)

        for file_path in files_to_upload:
            asset_name = os.path.basename(file_path)
            try:
                success, skipped, _ = _upload_asset_internal(release, file_path, asset_name)
                if success: upload_successful_count += 1
                if skipped: upload_skipped_count += 1
            except (RateLimitExceededException, ConnectionAbortedError) as rl_e:
                upload_failed = True; break
            except Exception as asset_e:
                upload_failed = True
                # break # Optionally break

        # 6. Final Upload Summary (Logic remains the same)
        print_log("STEP", "GitHub Upload Summary")
        failures = total_files - upload_successful_count - upload_skipped_count
        print_log("INFO", f"Assets Uploaded:       {upload_successful_count}")
        print_log("INFO", f"Assets Skipped:        {upload_skipped_count} (Already Existed)")
        print_log("INFO", f"Assets Failed:         {failures}")

        if upload_failed or failures > 0:
            print_log("ERROR", "One or more uploads failed. Release may be incomplete.")
            return None # Return None on failure
        else:
            if upload_successful_count + upload_skipped_count == total_files:
                print_log("SUCCESS", f"All {total_files} expected assets uploaded or confirmed existing.")
                print_log("INFO", f"Release URL: {release_url}")
                return github_info # Return collected info on success
            else:
                print_log("ERROR", f"Upload count mismatch. Expected {total_files}, accounted for {upload_successful_count + upload_skipped_count}.")
                return None # Return None on mismatch

    # --- Exception Handling (Include new sanitization errors if any, but mostly same) ---
    except (RateLimitExceededException, GithubException, FileNotFoundError, ValueError, RuntimeError, ConnectionAbortedError, Exception) as e:
        print_log("ERROR", f"GitHub upload process failed: {e}")
        if isinstance(e, GithubException) and not isinstance(e, (RateLimitExceededException, ConnectionAbortedError)):
             print_log("DEBUG", f"GitHub API Error Status: {e.status}, Data: {e.data}")
        elif not isinstance(e, (FileNotFoundError, ValueError, RuntimeError, RateLimitExceededException, ConnectionAbortedError)):
             traceback.print_exc()
        return None # Explicitly return None on any error
# --- Main Orchestration Functions ---
def perform_processing_or_encryption(args: argparse.Namespace, mode: str):
    """Handles 'process' and 'encrypt' modes."""
    # --- Setup (mostly the same) ---
    source_file_path = args.source_file
    part_size_bytes = args.part_size * 1024 * 1024
    hash_buffer_size = args.buffer_size * 1024 * 1024
    max_workers = args.workers
    do_github_upload = args.upload_to_github

    if mode == "process":
        task_function = process_plaintext_part
        output_dir_template = DEFAULT_PROC_OUTPUT_DIR_TEMPLATE
        upload_json_on_github = True
        mode_display_name = "PROCESS (Split)"
    elif mode == "encrypt":
        task_function = process_encryption_part
        output_dir_template = DEFAULT_ENC_OUTPUT_DIR_TEMPLATE
        upload_json_on_github = False
        mode_display_name = "ENCRYPTION"
        if not CRYPTOGRAPHY_AVAILABLE:
            print_log("ERROR", f"Cryptography library required for '{mode}' mode.")
            sys.exit(1)
    else: raise ValueError(f"Internal error: Invalid mode '{mode}'")

    start_process_time = time.monotonic()
    print_log("STEP", f"Initiating {mode_display_name} for: {source_file_path}")
    print_log("INFO", f"Config: Part Size={args.part_size}MB, Workers={max_workers}, Hash Buffer={args.buffer_size}MB")
    if do_github_upload:
        if not PYGITHUB_AVAILABLE:
            print_log("ERROR", "PyGithub library required for GitHub upload.")
            sys.exit(1)
        json_action = "including" if upload_json_on_github else "excluding"
        print_log("INFO", f"{Fore.YELLOW}GitHub upload requested ({json_action} metadata JSON).{Style.RESET_ALL}")

    # --- Init Vars (github_upload_info is the important one here) ---
    gh: Optional[Github] = None; repo: Optional[Repository] = None
    github_repo_name: Optional[str] = None; github_token: Optional[str] = None
    original_sha256: Optional[str] = None; original_filename: str = "N/A"
    file_size: int = -1
    github_upload_info: Optional[Dict[str, str]] = None # Will hold repo, orig_tag, sanitized_tag

    # --- GitHub Pre-Check (If Uploading) ---
    if do_github_upload:
        print_log("STEP", "Preparing for GitHub Check/Upload")
        try:
            print_log("INFO", "Getting GitHub credentials...")
            github_repo_name, github_token = get_github_credentials(prompt_for_repo=True)
            if not github_repo_name or not github_token:
                raise ValueError("GitHub repository and token required for upload/check.")

            gh = _get_github_instance(github_token) # Verifies token
            repo = _get_repo(gh, github_repo_name) # Verifies repo access

            print_log("INFO", "Calculating original file SHA for GitHub check...")
            if not os.path.isfile(source_file_path): raise FileNotFoundError(f"Source file not found: {source_file_path}")
            file_size = os.path.getsize(source_file_path)
            original_filename = os.path.basename(source_file_path)
            if not original_filename: raise ValueError("Source file path invalid.")
            original_sha256 = calculate_sha256(source_file_path, file_size, hash_buffer_size, "Original SHA (Check)")
            if not original_sha256: raise RuntimeError("Failed to calculate original file SHA.")
            print_log("INFO", f"Original file SHA-256 (for check): {Fore.GREEN}{original_sha256}{Style.RESET_ALL}")

            if repo: # If repo was found during access check
                print_log("STEP", "Checking GitHub for existing release...")
                original_sha256_short = original_sha256[:8]
                
                # --- MODIFICATION: Check using the *sanitized* version of the tag ---
                proposed_tag_for_check = f"file-{mode}-{original_filename}-{original_sha256_short}"
                expected_sanitized_tag = sanitize_release_version_tag(proposed_tag_for_check)
                print_log("INFO", f"Checking for sanitized tag: '{expected_sanitized_tag}'")
                exists, release_obj = _check_release_exists_internal(repo, expected_sanitized_tag)
                # --- END MODIFICATION ---

                if exists:
                    print_log("WARN", f"{Fore.YELLOW}SKIP: Release matching file/mode already exists (using sanitized tag '{expected_sanitized_tag}').{Style.RESET_ALL}")
                    release_url = release_obj.html_url if release_obj else "N/A"
                    created_at = release_obj.created_at if release_obj else "N/A"
                    num_assets = 'N/A'
                    if release_obj:
                        try: num_assets = len(list(release_obj.get_assets()))
                        except Exception as asset_err: print_log("DEBUG", f"Could not get asset count: {asset_err}")
                    print_log("WARN", f"  Repository:  {github_repo_name}")
                    print_log("WARN", f"  Release Tag: {expected_sanitized_tag}") # Show the tag found
                    print_log("WARN", f"  Release URL: {release_url}")
                    print_log("WARN", f"  Created At:  {created_at}")
                    print_log("WARN", f"  Assets Found:{num_assets}")
                    print_log("SUCCESS", "Operation skipped.")
                    sys.exit(0) # Exit cleanly if release already exists
                else:
                    print_log("INFO", "No matching release found. Proceeding...")
            else: # Repo was None from _get_repo
                 print_log("INFO", f"Repository '{github_repo_name}' not found. Will attempt creation later.")

        # --- Exception Handling (remains the same) ---
        except (RateLimitExceededException, ConnectionAbortedError) as rl_e:
            print_log("ERROR", f"GitHub API Error during pre-check: {rl_e}")
            sys.exit(1)
        except (ValueError, FileNotFoundError, RuntimeError, ConnectionError, GithubException, Exception) as e:
            print_log("ERROR", f"Error during GitHub pre-check: {e}")
            if isinstance(e, GithubException): print_log("DEBUG", f"Status: {e.status}, Data: {e.data}")
            elif not isinstance(e, (ValueError, FileNotFoundError, RuntimeError, ConnectionError)): traceback.print_exc()
            print_log("ERROR", "Aborting due to pre-check failure.")
            sys.exit(1)

    # --- Local File Processing (Steps 1-6: Largely the same) ---
    processing_successful = False
    output_dir = None; json_filepath = None
    metadata: Optional[Dict[str, Any]] = None

    try:
        # 1. Final File Validation & SHA (Logic remains the same)
        print_log("INFO", "Validating source file and preparing...")
        # ... (validation logic as before) ...
        if file_size == -1: # If not done during pre-check
            if not os.path.isfile(source_file_path): raise FileNotFoundError(f"Source file not found: {source_file_path}")
            file_size = os.path.getsize(source_file_path)
            original_filename = os.path.basename(source_file_path)
            if not original_filename: raise ValueError("Source path yields no filename.")
            print_log("STEP", "Calculating SHA-256 checksum of original file")
            original_sha256 = calculate_sha256(source_file_path, file_size, hash_buffer_size, "Original File Hash")
            if not original_sha256: raise RuntimeError("Failed to calculate original file SHA.")
            print_log("INFO", f"Original file SHA-256: {Fore.GREEN}{original_sha256}{Style.RESET_ALL}")
        elif original_sha256 is None or original_filename == "N/A":
             raise RuntimeError("File metadata (SHA/Name) missing after pre-check.")
        else: # Already validated
             print_log("INFO", f"Using pre-validated info: '{original_filename}', Size: {format_bytes(file_size)}, SHA: {Fore.GREEN}{original_sha256[:12]}...{Style.RESET_ALL}")
        
        original_filename_base = os.path.splitext(original_filename)[0] if original_filename else "file"

        # 2. Calculate Number of Parts (Logic remains the same)
        if file_size == 0: num_parts = 0; print_log("WARN", "Source file is empty.")
        else:
            if part_size_bytes <= 0: raise ValueError("Part size must be positive.")
            num_parts = math.ceil(file_size / part_size_bytes)
            if part_size_bytes > 1.9 * 1024 * 1024 * 1024:
                 print_log("WARN", f"Target part size ({args.part_size} MB) large, may exceed GitHub limit (~2 GiB).")
        print_log("INFO", f"Calculated number of parts: {num_parts}")

        # 3. Prepare Output Directory (Logic remains the same)
        output_dir_name = args.output_dir or output_dir_template.format(original_filename_base)
        output_dir = os.path.abspath(output_dir_name)
        print_log("STEP", f"Preparing output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        print_log("INFO", f"Output directory ensured: '{output_dir}'")
        json_filename = f"{original_sha256}.json"
        json_filepath = os.path.join(output_dir, json_filename)
        print_log("INFO", f"Metadata JSON file will be: '{json_filename}'")

        # 4. Initialize Metadata Dictionary (Logic remains the same)
        metadata = {
            "processor_version": PROCESSOR_VERSION, "mode": mode,
            "original_filename": original_filename, "original_sha256": original_sha256,
            "file_size_bytes": file_size, "part_size_bytes": part_size_bytes,
            "number_of_parts": num_parts, "parts": [],
            "creation_datetime_utc": None, "total_processing_time": None,
            "total_processing_seconds": None,
            "settings": {
                "part_size_mb": args.part_size, "workers": max_workers, "hash_buffer_mb": args.buffer_size,
                "output_directory": output_dir, # Absolute path
                "github_upload_requested": do_github_upload,
                "github_repo": github_repo_name, # Store repo name if provided
                # Tags added to DB later via github_info
            }
        }

        # 5. Parallel Part Processing (Logic remains the same)
        start_parts_processing_time = time.monotonic()
        if num_parts == 0:
            print_log("INFO", "Skipping part processing (empty file).")
            all_parts_metadata = []
            processing_successful = True
        else:
            print_log("STEP", f"Starting parallel {mode_display_name} ({max_workers} workers)")
            # ... (task creation and execution logic as before) ...
            tasks = []
            processed_bytes = 0
            for i in range(1, num_parts + 1):
                part_start_offset = processed_bytes
                bytes_in_part = min(part_size_bytes, file_size - processed_bytes)
                if bytes_in_part <= 0: break
                tasks.append({
                    'part_number': i, 'source_path': source_file_path, 'offset': part_start_offset,
                    'size': bytes_in_part, 'output_dir': output_dir,
                    'hash_buffer_size': hash_buffer_size, 'total_parts': num_parts,
                })
                processed_bytes += bytes_in_part

            all_parts_metadata = []
            futures: List[Future] = []
            encountered_worker_error = False
            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=f'{mode[:3].capitalize()}Worker') as executor:
                print_log("INFO", f"Submitting {len(tasks)} {mode} tasks...")
                futures = [executor.submit(task_function, task_info) for task_info in tasks]
                print_log("INFO", f"Waiting for {mode} parts to complete...")
                try:
                    for future in tqdm(as_completed(futures), total=len(futures), unit='part',
                                       desc=f"{Fore.YELLOW}{mode_display_name.capitalize()} Parts{Style.RESET_ALL}",
                                       bar_format=TQDM_BAR_FORMAT, ncols=90, leave=False):
                         try:
                             part_result = future.result()
                             if part_result: all_parts_metadata.append(part_result)
                         except Exception as e:
                             print_log("ERROR", f"A {mode} worker thread failed: {e}")
                             encountered_worker_error = True
                except KeyboardInterrupt:
                     print_log("WARN", f"\nKeyboardInterrupt during {mode}. Attempting shutdown...")
                     encountered_worker_error = True
                     for f in futures: f.cancel()
                     time.sleep(1)
                     print_log("WARN", f"{mode_display_name} cancelled by user.")

            if encountered_worker_error:
                 raise RuntimeError(f"{mode_display_name} failed or was cancelled.")
            if len(all_parts_metadata) != num_parts:
                 raise RuntimeError(f"Inconsistent state: Expected {num_parts} parts, collected metadata for {len(all_parts_metadata)}.")

            processing_successful = True
            all_parts_metadata.sort(key=lambda p: p['part_number'])
            metadata["parts"] = all_parts_metadata
            metadata["number_of_parts"] = len(all_parts_metadata) # Update just in case


        # 6. Finalize Metadata and Write JSON (Logic remains the same)
        if processing_successful:
            end_parts_processing_time = time.monotonic()
            total_processing_time_seconds = end_parts_processing_time - start_parts_processing_time
            metadata["creation_datetime_utc"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            metadata["total_processing_time"] = format_duration(total_processing_time_seconds)
            metadata["total_processing_seconds"] = round(total_processing_time_seconds, 2)

            print_log("STEP", f"Writing metadata to JSON: {json_filepath}")
            try:
                with open(json_filepath, 'w') as jsonfile:
                    json.dump(metadata, jsonfile, indent=4)
                print_log("SUCCESS", f"Successfully wrote metadata to '{json_filepath}'")
            except Exception as e:
                raise RuntimeError(f"Could not save metadata file '{json_filepath}': {e}") from e
        else:
             raise RuntimeError("Processing unsuccessful before JSON write.")

        # 7. GitHub Upload Step (Conditional - Calls updated upload function)
        # --- MODIFICATION: github_upload_info now contains original/sanitized tags ---
        github_upload_info = None # Reset before attempt
        if do_github_upload and processing_successful:
            if not gh or not github_repo_name:
                 # Get creds now if pre-check was skipped (logic same)
                 if not gh:
                     print_log("INFO", "Getting GitHub credentials for upload...")
                     github_repo_name, github_token = get_github_credentials(prompt_for_repo=True)
                     if not github_repo_name or not github_token:
                         raise ValueError("GitHub repo and token needed for upload.")
                     gh = _get_github_instance(github_token)
                 elif not github_repo_name:
                     raise ValueError("GitHub repo name missing despite instance existing.")

            # Re-check repo existence if needed (logic same)
            if not repo:
                repo = _get_repo(gh, github_repo_name)
                # Creation handled inside upload func if still None

            # Call the updated upload function
            github_upload_info = upload_to_github_release(
                gh=gh, output_dir=output_dir, metadata=metadata,
                repo_full_name=github_repo_name,
                upload_json_file=upload_json_on_github
            )
            # github_upload_info now contains: repo_full_name, original_release_tag, sanitized_release_tag if successful
            if github_upload_info is None:
                 print_log("WARN", "GitHub upload failed or was incomplete.")
        # --- END MODIFICATION ---

        # 8. Add Log Entry to Database (via Server)
        # --- MODIFICATION: Pass the full github_upload_info dictionary ---
        if processing_successful and metadata:
            # Pass the entire github_upload_info dict obtained from upload_to_github_release
            # It will be None if upload wasn't requested or failed.
            db_added = db_add_log_entry(metadata, github_info=github_upload_info) 
            if not db_added:
                 print_log("WARN", "Failed to add log entry to database via server.")
        # --- END MODIFICATION ---


        # 9. Final Summary Log (Update to show sanitized tag)
        end_process_time = time.monotonic()
        total_overall_time_seconds = end_process_time - start_process_time
        print_log("SUCCESS", f"\n--- {mode_display_name} Summary ---")
        print_log("INFO", f"Original file:         '{original_filename}'")
        print_log("INFO", f"Original SHA-256:      {original_sha256}")
        print_log("INFO", f"Output directory:      '{output_dir}'")
        print_log("INFO", f"Metadata file:         '{json_filename}'")
        if mode == "encrypt": print_log("INFO", f"                       {Fore.YELLOW}(Contains Keys){Style.RESET_ALL}")
        if metadata:
            print_log("INFO", f"Total parts created:   {metadata.get('number_of_parts', 'N/A')}")
            print_log("INFO", f"Processing time:       {metadata.get('total_processing_time', 'N/A')}")
        else: print_log("WARN","Metadata unavailable for summary.")

        # --- MODIFICATION: Update GitHub status display ---
        if do_github_upload:
            upload_status_str = f"{Fore.YELLOW}Not Attempted (Processing Error?){Style.RESET_ALL}"
            if github_upload_info: # Check if info dict exists (means upload attempt was made and potentially succeeded)
                 upload_status_str = f"{Fore.GREEN}Successful{Style.RESET_ALL}"
                 # Extract info from the dict
                 final_sanitized_tag = github_upload_info.get('sanitized_release_tag')
                 repo_name = github_upload_info.get('repo_full_name')
                 if final_sanitized_tag and repo_name:
                     release_url = f"https://github.com/{repo_name}/releases/tag/{final_sanitized_tag}"
                     print_log("INFO", f"GitHub Release URL:    {release_url}")
                     print_log("INFO", f"GitHub Sanitized Tag:  '{final_sanitized_tag}'") # Display sanitized tag
                 else:
                      # Should not happen if github_upload_info is populated correctly
                      print_log("WARN", "GitHub upload info present but missing tag/repo name.")
                      upload_status_str = f"{Fore.YELLOW}Successful (Info Missing){Style.RESET_ALL}"
            elif github_upload_info is None and processing_successful: # Upload attempted but failed (upload func returned None)
                 upload_status_str = f"{Fore.RED}Failed{Style.RESET_ALL}"
            print_log("INFO", f"GitHub Upload Status:  {upload_status_str}")
        # --- END MODIFICATION ---

        db_status_str = f"{Fore.GREEN}Logged via Server{Style.RESET_ALL}" if 'db_added' in locals() and db_added else f"{Fore.YELLOW}Logging Failed/Skipped{Style.RESET_ALL}"
        print_log("INFO", f"Database Log Status:   {db_status_str} (via {DB_SERVER_URL})")
        print_log("INFO", f"Total Overall Time:    {format_duration(total_overall_time_seconds)}")
        print_log("SUCCESS", "-------------------------------------")

    # --- Exception Handling (remains the same) ---
    except (MemoryError, ValueError, RuntimeError, OSError, FileNotFoundError,
            GithubException, ConnectionError, ConnectionAbortedError, requests.exceptions.RequestException, Exception) as e:
        print_log("ERROR", f"\nA critical error occurred during {mode}: {e}")
        if json_filepath and os.path.exists(json_filepath) and not processing_successful:
            try: os.remove(json_filepath); print_log("INFO", f"Removed incomplete JSON '{json_filepath}'")
            except Exception as rm_e: print_log("WARN", f"Could not remove incomplete JSON: {rm_e}")

        if isinstance(e, GithubException) and not isinstance(e, (RateLimitExceededException, ConnectionAbortedError)):
             print_log("DEBUG", f"GitHub API Error Status: {e.status}, Data: {e.data}")
        elif isinstance(e, requests.exceptions.RequestException):
             print_log("DEBUG", f"DB Server Request Error: {e.request} -> {e.response}")
        elif not isinstance(e, (ValueError, RuntimeError, OSError, FileNotFoundError,
                                KeyboardInterrupt, ConnectionError, ImportError,
                                RateLimitExceededException, ConnectionAbortedError)):
             print(f"{Fore.RED}{traceback.format_exc()}{Style.RESET_ALL}")
        sys.exit(1)
# --- Wrapper functions ---
def process_main(args: argparse.Namespace):
    perform_processing_or_encryption(args, "process")

def encrypt_main(args: argparse.Namespace):
    perform_processing_or_encryption(args, "encrypt")

# --- GitHub Download Helpers (for Decrypt) ---

def _display_download_status(layer_status: Dict[str, str], status_lock: threading.Lock, stop_event: threading.Event):
    """Thread function to display dynamic download status."""
    num_layers = len(layer_status)
    if num_layers == 0: return # Nothing to display

    try:
        terminal_width = shutil.get_terminal_size().columns
    except OSError:
        terminal_width = 80 # Fallback

    sorted_layers = sorted(layer_status.keys(), key=lambda x: int(x.split('_')[-1])) # Sort by part number

    # Initial display - print all lines
    with status_lock:
        for layer_name in sorted_layers:
            status = layer_status.get(layer_name, LAYER_WAITING)
            print(f"{layer_name}: {status}".ljust(terminal_width))
    sys.stdout.flush()

    while not stop_event.is_set():
        time.sleep(0.15) # Update frequency

        # Move cursor up N lines
        print(f"\033[{num_layers}F", end="")

        with status_lock:
            for layer_name in sorted_layers:
                status = layer_status.get(layer_name, LAYER_WAITING)
                # Clear line and print updated status
                print(f"\033[K{layer_name}: {status}".ljust(terminal_width))
        sys.stdout.flush()

    # Final update after stopping to ensure correct final state is shown
    print(f"\033[{num_layers}F", end="")
    with status_lock:
        for layer_name in sorted_layers:
            status = layer_status.get(layer_name, "Unknown")
            print(f"\033[K{layer_name}: {status}".ljust(terminal_width))
    sys.stdout.flush()
    print_log("INFO", "Download display thread finished.")


def find_or_prompt_aria2c() -> Optional[str]:
    """
    Finds the aria2c executable path based on the OS or prompts the user to install it.

    Returns:
        The path to the aria2c executable if found, otherwise None.
    """
    system = platform.system()
    print_log("INFO", f"Searching for aria2c executable (OS: {system})...")

    aria2c_path = None
    if system == "Windows":
        # Look in a 'drivers' subdirectory relative to the script's CWD
        expected_path = os.path.join(os.getcwd(), 'drivers', os.getenv("DRIVER_NAME" , "aria2c.exe"))
        if os.path.isfile(expected_path):
            aria2c_path = expected_path
            print_log("INFO", f"Found aria2c on Windows: {aria2c_path}")
        else:
             print_log("WARN", f"aria2c.exe not found at expected Windows location: {expected_path}")
             print_log("WARN", "Please ensure 'aria2c.exe' is placed in a 'drivers' folder in the script's directory.")

    elif system == "Linux" or system == "Darwin": # Linux or macOS
        aria2c_path = shutil.which("aria2c")
        if aria2c_path:
            print_log("INFO", f"Found aria2c in PATH: {aria2c_path}")
        else:
            print_log("ERROR", "'aria2c' command not found in system PATH.")
            distro = platform.system() # Could add more specific distro detection if needed
            install_cmd = "sudo apt update && sudo apt install aria2" if "Linux" in distro else "brew install aria2" if system == "Darwin" else "your_package_manager install aria2"
            print_log("ERROR", f"Please install aria2c manually (e.g., using: '{install_cmd}') and ensure it's in your PATH.")
            print_log("WARN", "Attempting automatic installation via apt (requires sudo)...")
            try:
                password = getpass.getpass(f"{Fore.YELLOW}--> Enter SUDO password to install aria2c:{Style.RESET_ALL} ")
                update_cmd = f'echo "{password}" | sudo -S apt-get update'
                install_cmd = f'echo "{password}" | sudo -S apt-get install -y aria2c'
                print_log("INFO","Running apt update...")
                update_proc = subprocess.run(update_cmd, shell=True, capture_output=True, text=True)
                if update_proc.returncode != 0: print_log("ERROR", f"apt update failed: {update_proc.stderr}")
                else: print_log("INFO", "Running apt install aria2c...")
                install_proc = subprocess.run(install_cmd, shell=True, capture_output=True, text=True)
                if install_proc.returncode == 0: aria2c_path = shutil.which("aria2c"); print_log("SUCCESS", "aria2c installed.")
                else: print_log("ERROR", f"aria2c installation failed: {install_proc.stderr}")
            except Exception as e: print_log("ERROR", f"Error during automatic install attempt: {e}")
            finally: password = "" # Clear password var

    else:
        print_log("WARN", f"Unsupported OS '{system}' for automatic aria2c path detection.")

    if not aria2c_path:
        print_log("ERROR", "aria2c executable could not be found or installation failed/skipped.")
        return None

    # Optional: Verify it's executable (though shutil.which usually does this)
    if not os.access(aria2c_path, os.X_OK):
         print_log("ERROR", f"Found aria2c at '{aria2c_path}', but it is not executable.")
         return None

    return aria2c_path

# --- Updated: download_asset_parallel (Now uses aria2c) ---

def download_asset_parallel(task_info: Dict[str, Any]) -> Tuple[str, bool, bool]:
    """
    Downloads a single asset using aria2c subprocess, verifies SHA, updates status.
    Returns (asset_name, download_success, sha_match).
    """
    asset: GitReleaseAsset = task_info['asset']
    download_dir = task_info['download_dir']
    token = task_info['token'] # Needed for API URL access, aria2c *might* need it via --header
    aria2c_path = task_info['aria2c_path'] # Path to aria2c executable
    max_retries = 1 # Let aria2c handle internal retries if configured; simplify python loop
    layer_status = task_info['layer_status']
    status_lock = task_info['status_lock']
    expected_encrypted_sha = task_info['expected_encrypted_sha'] # SHA of the asset on GitHub
    hash_buffer_size = task_info['hash_buffer_size']

    asset_name = asset.name
    asset_size = asset.size # Use for final size check
    # API URL usually redirects to the actual download URL (e.g., objects.githubusercontent.com)
    # aria2c typically handles redirects well.
    api_download_url = asset.url
    if not api_download_url:
        print_log("ERROR", f"Could not get API URL (asset.url) for asset {asset_name}. Cannot download.")
        with status_lock: layer_status[asset_name] = LAYER_PULL_FAILED
        return asset_name, False, False

    # Ensure download directory exists
    os.makedirs(download_dir, exist_ok=True)
    download_path = os.path.join(download_dir, asset_name) # Final expected path
    part_num_str = asset_name.split('_')[-1] if asset_name.startswith('part_') else None

    def _update_status(new_status):
        with status_lock:
            layer_status[asset_name] = new_status

    _update_status(f"{LAYER_PULLING} (aria2c)") # Update status message

    # --- Prepare aria2c Command ---
    aria2c_options = [
        "-x16",                 # Max connections per server
        "-s16",                 # Min split size
        "--summary-interval=0", # Disable live summary to hide default progress (use 1 for debug)
        "-j5",                  # Max concurrent downloads (aria2c internal)
        "--disk-cache=0",       # Disable disk cache
        "--min-split-size=22M" , 
        "--file-allocation=none",# Allocation mode
        "--enable-mmap=true",   # Use memory map if possible2
        "--quiet=true",         # Suppress most output except errors
        "--show-console-readout=false", # Hide live speed/ETA
        # "--console-log-level=warn", # Further reduce log level if needed
        f"--dir={download_dir}", # Set download directory
        f"--out={asset_name}",   # Set output filename within the directory
    ]
    # Add header for potentially accessing private repo assets via API URL redirect
    # Note: The token gives access to the API URL, which then provides a short-lived,
    # pre-signed URL to the actual object storage. aria2c *should* follow this
    # redirect and not need the token for the final download URL itself.
    # However, including it might help if GitHub's redirect mechanism changes.
    if token:
        aria2c_options.append(f"--header=Authorization: token {token}")
        # Also need Accept header for the initial API call
        aria2c_options.append(f"--header=Accept: application/octet-stream")

    command = [aria2c_path] + aria2c_options + [api_download_url]

    download_success = False
    sha_match = False
    retries = 0 # Python-level retry logic removed for simplicity for now

    # --- Execute aria2c ---
    print_log("DEBUG", f"Executing aria2c for {asset_name}: {' '.join(command)}", part_num=part_num_str)
    try:
        # Use subprocess.run to wait for completion
        process = subprocess.run(
            command,
            capture_output=True, # Capture stdout/stderr
            text=True,           # Decode output as text
            check=False          # Don't raise exception on non-zero exit (we check manually)
        )

        if process.returncode == 0:
            print_log("DEBUG", f"aria2c completed successfully for {asset_name}.", part_num=part_num_str)
            download_success = True
        else:
            print_log("ERROR", f"aria2c failed for {asset_name} (Exit Code: {process.returncode}).", part_num=part_num_str)
            print_log("ERROR", f"aria2c stderr: {process.stderr.strip()}", part_num=part_num_str)
            if process.stdout: # Log stdout too in case of errors
                print_log("ERROR", f"aria2c stdout: {process.stdout.strip()}", part_num=part_num_str)
            download_success = False
            _update_status(LAYER_PULL_FAILED)
            # Clean up potentially partial file
            if os.path.exists(download_path):
                try: os.remove(download_path)
                except OSError: pass
            return asset_name, False, False # Exit function on aria2c error

    except FileNotFoundError:
        print_log("ERROR", f"Failed to execute aria2c. Path not found or invalid: {aria2c_path}")
        _update_status(LAYER_PULL_FAILED)
        return asset_name, False, False
    except Exception as e:
        print_log("ERROR", f"Unexpected error executing aria2c for {asset_name}: {e}", part_num=part_num_str)
        _update_status(LAYER_PULL_FAILED)
        return asset_name, False, False

    # --- Verification after successful aria2c download ---
    if download_success:
        try:
            if not os.path.isfile(download_path):
                raise FileNotFoundError(f"aria2c reported success, but output file '{download_path}' is missing!")

            downloaded_size = os.path.getsize(download_path)

            # Check size against GitHub metadata size (asset_size)
            if downloaded_size != asset_size:
                  # Warning, but proceed to SHA check as aria2c might handle compressed downloads etc.
                  print_log("WARN", f"Size mismatch for {asset_name}. GH Meta: {asset_size}, Actual: {downloaded_size}. Proceeding with SHA check.", part_num=part_num_str)
                  # Note: For very large files, GitHub might use LFS or other mechanisms where size reported might differ. SHA is the ultimate check.

            _update_status(LAYER_VERIFYING_DL)

            if expected_encrypted_sha: # Only verify SHA for encrypted mode downloads
                actual_sha = calculate_sha256(download_path, downloaded_size, hash_buffer_size,
                                              desc_prefix=f"Verify DL {asset_name}", leave=False, position=int(part_num_str or 0)+1)

                if actual_sha and actual_sha == expected_encrypted_sha:
                    sha_match = True
                    _update_status(LAYER_DOWNLOADED)
                    print_log("INFO", f"SHA verified successfully for {asset_name}.", part_num=part_num_str)
                elif actual_sha:
                    sha_match = False
                    _update_status(LAYER_SHA_MISMATCH_DL)
                    print_log("ERROR", f"SHA Mismatch after download for {asset_name}. Expected {expected_encrypted_sha[:8]}.., Got {actual_sha[:8]}..", part_num=part_num_str)
                else: # Hash calculation failed
                    sha_match = False
                    _update_status(LAYER_PULL_FAILED) # Treat hash fail as pull fail
                    print_log("ERROR", f"Hash calculation failed after download for {asset_name}", part_num=part_num_str)
                    raise Exception("SHA256 calculation failed post-download.")

            else: # 'process' mode, no encrypted SHA from metadata
                sha_match = True # Assume download OK if aria2c succeeded and size is plausible
                _update_status(LAYER_DOWNLOADED)
                print_log("INFO", f"Download complete (process mode, no SHA verification) for {asset_name}.", part_num=part_num_str)

        except (FileNotFoundError, IOError, Exception) as verify_e:
             print_log("ERROR", f"Verification failed for {asset_name} after aria2c success: {verify_e}", part_num=part_num_str)
             download_success = False # Mark overall success as False
             sha_match = False
             _update_status(LAYER_PULL_FAILED)
             # Clean up file if verification fails
             if os.path.exists(download_path):
                try: os.remove(download_path)
                except OSError: pass

    # Final result: success is True only if download AND verification passed
    overall_success = download_success and sha_match
    if not overall_success and layer_status.get(asset_name) != LAYER_PULL_FAILED and layer_status.get(asset_name) != LAYER_SHA_MISMATCH_DL:
        # Ensure status reflects failure if not already set
         _update_status(LAYER_PULL_FAILED)

    return asset_name, overall_success, sha_match # Return overall success flag


# --- Updated: download_parts_from_github (Calls find_or_prompt_aria2c and passes path) ---

def download_parts_from_github(repo_full_name: str, release_tag: str, metadata: Dict[str, Any], download_workers: int, hash_buffer_size: int) -> Optional[str]:
    """Downloads required parts using aria2c. Returns temp dir path or None."""
    if not PYGITHUB_AVAILABLE: print_log("ERROR", "PyGithub needed for download."); return None
    # aria2c check is now done inside this function

    print_log("STEP", f"Attempting download via aria2c from GitHub Repo: {repo_full_name}, Tag: {release_tag}")

    # --- MODIFICATION: Find aria2c first ---
    aria2c_exec_path = find_or_prompt_aria2c()
    if not aria2c_exec_path:
        print_log("ERROR", "Cannot proceed with download without a valid aria2c executable.")
        return None
    # --- END MODIFICATION ---

    token = None; gh = None; repo = None
    try:
        _, token = get_github_credentials(prompt_for_repo=False) # Only need token
        if not token: raise ValueError("GitHub token required for release access.")
        gh = _get_github_instance(token) # Verifies token
        repo = _get_repo(gh, repo_full_name) # Verifies repo access
        if not repo: raise ValueError(f"Repository '{repo_full_name}' not found/inaccessible.")
    except (ValueError, ConnectionError, ConnectionAbortedError, GithubException) as cred_e:
        print_log("ERROR", f"GitHub access failed: {cred_e}")
        return None

    temp_download_dir = None
    status_lock = threading.Lock()
    stop_display_event = threading.Event()
    layer_status: Dict[str, str] = {}
    display_thread = None

    try:
        print_log("INFO", f"Fetching release with tag '{release_tag}'...")
        release = repo.get_release(release_tag)
        print_log("INFO", f"Found release '{release.title}' created at {release.created_at}.")

        original_filename_base = os.path.splitext(metadata['original_filename'])[0]
        sha_short = metadata['original_sha256'][:8]
        temp_dir_name = DEFAULT_DECRYPT_TEMP_DIR_TEMPLATE.format(original_filename_base, sha_short)
        temp_download_dir = os.path.abspath(temp_dir_name)

        if os.path.exists(temp_download_dir):
             print_log("WARN", f"Temp dir '{temp_download_dir}' exists. Removing.")
             try: shutil.rmtree(temp_download_dir)
             except OSError as e: print_log("ERROR", f"Failed removing existing temp dir: {e}"); return None
        os.makedirs(temp_download_dir, exist_ok=True)
        print_log("INFO", f"Created temporary download directory: '{temp_download_dir}'")

        print_log("INFO", "Identifying required layers...")
        try: assets = list(release.get_assets()); asset_map = {asset.name: asset for asset in assets}
        except Exception as e: raise ValueError(f"Could not get assets from release: {e}") from e

        parts_metadata = metadata.get('parts', [])
        required_layers: Dict[str, Dict[str, Any]] = {}
        missing_layers = set(); missing_shas = set()
        for p_meta in parts_metadata:
             part_num = p_meta['part_number']
             layer_filename = p_meta.get("part_filename") or f"part_{part_num}"
             expected_sha = None
             if metadata['mode'] == 'encrypt':
                 expected_sha = p_meta.get("encrypted_sha256")

             asset_obj = asset_map.get(layer_filename)
             if asset_obj:
                 if metadata['mode'] == 'encrypt' and not expected_sha: missing_shas.add(layer_filename)
                 required_layers[layer_filename] = {"asset": asset_obj, "sha": expected_sha}
                 layer_status[layer_filename] = LAYER_WAITING # Initialize status
             else:
                 missing_layers.add(layer_filename)

        if missing_layers: raise FileNotFoundError(f"Missing required layer assets on release: {', '.join(sorted(list(missing_layers)))}")
        if missing_shas: raise ValueError(f"Missing 'encrypted_sha256' in metadata for layers (needed for verification): {', '.join(sorted(list(missing_shas)))}")
        if not required_layers and metadata['number_of_parts'] > 0: raise ValueError("No layers to download, but parts required.")
        if not required_layers and metadata['number_of_parts'] == 0: print_log("INFO", "No layers required (empty file)."); return temp_download_dir

        num_layers_to_pull = len(required_layers)
        # Note: download_workers now controls how many aria2c processes run concurrently
        actual_download_workers = min(download_workers, MAX_DOWNLOAD_WORKERS, num_layers_to_pull if num_layers_to_pull > 0 else 1)
        print_log("STEP", f"Starting parallel pull of {num_layers_to_pull} layers via aria2c ({actual_download_workers} concurrent processes)...")

        display_thread = threading.Thread(target=_display_download_status, args=(layer_status, status_lock, stop_display_event), daemon=True); display_thread.start()

        # --- MODIFICATION: Pass aria2c_exec_path to tasks ---
        download_tasks = [{'asset': layer_info['asset'],
                           'download_dir': temp_download_dir,
                           'token': token,
                           'aria2c_path': aria2c_exec_path, # <-- Pass path
                           'layer_status': layer_status,
                           'status_lock': status_lock,
                           'expected_encrypted_sha': layer_info['sha'],
                           'hash_buffer_size': hash_buffer_size}
                          for layer_filename, layer_info in required_layers.items()]
        # --- END MODIFICATION ---

        pull_successful_count = 0; pull_failed_count = 0
        futures: List[Future] = []
        try:
            with ThreadPoolExecutor(max_workers=actual_download_workers, thread_name_prefix='PullWorker') as executor:
                futures = [executor.submit(download_asset_parallel, task) for task in download_tasks] # Calls the NEW download_asset_parallel
                for future in as_completed(futures):
                    try:
                        _, success, sha_verified = future.result() # Unpack result
                        # Overall success requires download AND SHA verification
                        if success and sha_verified: # Check was already done in download_asset_parallel
                            pull_successful_count += 1
                        else:
                            pull_failed_count += 1
                    except Exception as e:
                        print_log("ERROR", f"Download monitoring error: {e}")
                        pull_failed_count += 1 # Assume failure

        except KeyboardInterrupt:
            print_log("WARN", "\nPull cancelled by user. Cleaning up...")
            # Attempt to terminate running aria2c processes? Difficult without Popen objects.
            # Rely on ThreadPoolExecutor cancellation for now.
            for f in futures: f.cancel()
            time.sleep(1)
            raise KeyboardInterrupt("Pull cancelled") from None
        finally:
            stop_display_event.set()
            if display_thread: display_thread.join(timeout=2)

        print_log("INFO", f"Pull Summary: {pull_successful_count} Succeeded, {pull_failed_count} Failed.")
        if pull_failed_count > 0: raise RuntimeError(f"Failed to pull/verify {pull_failed_count} required layer(s) using aria2c.")

        print_log("SUCCESS", f"All layers pulled successfully via aria2c to '{temp_download_dir}'.")
        return temp_download_dir

    # --- Exception Handling (remains largely the same) ---
    except (RateLimitExceededException, ConnectionAbortedError) as rl_e:
        print_log("ERROR", f"GitHub API error during download setup: {rl_e}")
    except (ValueError, FileNotFoundError, RuntimeError, GithubException, requests.exceptions.RequestException, KeyboardInterrupt, Exception) as e:
        print_log("ERROR", f"Failed during GitHub download process: {e}")
        if not isinstance(e, (ValueError, FileNotFoundError, RuntimeError, KeyboardInterrupt, RateLimitExceededException, ConnectionAbortedError)):
             traceback.print_exc()
    finally:
        stop_display_event.set()
        if display_thread and display_thread.is_alive(): display_thread.join(timeout=1)

    # Cleanup temp dir on any error
    if temp_download_dir and os.path.exists(temp_download_dir):
        try:
            print_log("INFO", f"Cleaning up failed download directory: {temp_download_dir}")
            shutil.rmtree(temp_download_dir)
        except OSError as rm_e:
            print_log("WARN", f"Failed to clean up temp directory '{temp_download_dir}': {rm_e}")
    return None # Indicate failure


# --- Decryption / Reconstruction Functionality ---
def reconstruct_plaintext_file(metadata: Dict[str, Any], source_directory: str, output_file_path: str, hash_buffer_size: int) -> bool:
    """Reconstructs a file from plaintext parts ('process' mode)."""
    print_log("STEP", f"Combining layers from: {source_directory}")
    num_parts = metadata["number_of_parts"]; expected_size = metadata["file_size_bytes"]; original_sha256 = metadata["original_sha256"]
    if num_parts == 0:
        print_log("INFO", "Reconstructing empty file (0 layers).")
        try:
            with open(output_file_path, 'wb') as outfile: pass; empty_sha = hashlib.sha256(b'').hexdigest()
            if os.path.getsize(output_file_path) == 0 and empty_sha == original_sha256: print_log("SUCCESS", "Empty file reconstructed and verified."); return True
            else: print_log("ERROR", f"Empty file creation/SHA mismatch! Expected {original_sha256[:8]}..., Got {empty_sha[:8]}..."); return False
        except Exception as e: print_log("ERROR", f"Failed creating empty output file: {e}"); return False

    print_log("INFO", f"Verifying presence of {num_parts} plaintext layer files...")
    all_parts_found = True; required_files = []
    parts_metadata = metadata.get("parts", []) # Use metadata if available <-- ADDED LINE
    for i in range(1, num_parts + 1):
        # --- MODIFIED FILENAME LOGIC ---
        # Try to get specific filename from metadata, fallback to generic name
        part_meta = next((p for p in parts_metadata if p.get('part_number') == i), None)
        part_filename = (part_meta.get('part_filename') if part_meta else None) or f"part_{i}"
        # --- END MODIFIED FILENAME LOGIC ---
        part_filepath = os.path.join(source_directory, part_filename)
        required_files.append(part_filepath)
        if not os.path.isfile(part_filepath): print_log("ERROR", f"Required layer missing: '{part_filename}' in '{source_directory}'"); all_parts_found = False
    if not all_parts_found: print_log("ERROR", "Cannot reconstruct due to missing layers."); return False

    print_log("INFO", "All layers found. Starting assembly...")
    total_bytes_written = 0; assembly_start_time = time.monotonic()
    try:
        with open(output_file_path, 'wb') as outfile, \
             tqdm(total=expected_size, unit='B', unit_scale=True, unit_divisor=1024,
                  desc=f"{Fore.YELLOW}Combining Layers{Style.RESET_ALL}", bar_format=TQDM_BAR_FORMAT, ncols=90, leave=False) as pbar:
            for i in range(1, num_parts + 1): # Iterate through part numbers
                 # --- MODIFIED FILENAME LOGIC ---
                 part_meta = next((p for p in parts_metadata if p.get('part_number') == i), None)
                 part_filename = (part_meta.get('part_filename') if part_meta else None) or f"part_{i}"
                 # --- END MODIFIED FILENAME LOGIC ---
                 part_filepath = os.path.join(source_directory, part_filename)
                 try:
                     part_size = os.path.getsize(part_filepath)
                     with open(part_filepath, 'rb') as infile:
                         # Use larger buffer for potentially faster copy
                         buffer = infile.read(hash_buffer_size)
                         while buffer:
                             outfile.write(buffer)
                             buffer = infile.read(hash_buffer_size)
                     total_bytes_written += part_size
                     pbar.update(part_size)
                 except Exception as copy_e: raise IOError(f"Error reading/writing layer {part_filename}: {copy_e}") from copy_e
        assembly_end_time = time.monotonic()
        print_log("SUCCESS", f"Layer assembly completed in {format_duration(assembly_end_time - assembly_start_time)}.")
        final_size = os.path.getsize(output_file_path)
        if final_size != expected_size: print_log("ERROR", f"Final size ({format_bytes(final_size)}) mismatch! Expected: {format_bytes(expected_size)}."); return False
        else: print_log("INFO", f"Final size matches expected ({format_bytes(final_size)})."); return True
    except (IOError, OSError, Exception) as e:
        print_log("ERROR", f"Layer reconstruction failed: {e}")
        if not isinstance(e, (IOError, OSError)): traceback.print_exc()
        if os.path.exists(output_file_path): 
            try: 
                os.remove(output_file_path)
            except OSError: 
                pass
        return False


# --- Table Display Helper ---

def _display_entries_table(entries: List[Dict], console: Console, is_trash_list: bool = False) -> set:
    """Displays a list of entries (active or trash) using Rich table or basic print. Returns set of displayed valid IDs."""
    if not entries:
        type_str = "trash" if is_trash_list else "active"
        print_log("INFO", f"No {type_str} entries found.")
        return set() # Return empty set if nothing displayed

    use_rich = RICH_AVAILABLE # Use module-level check
    title = "Trashed Database Entries" if is_trash_list else "Available Database Entries"
    valid_ids = set()

    if use_rich:
        try:
            table = Table(title=title, show_header=True, header_style="bold magenta",
                          border_style="blue", show_lines=True, expand=True)
            table.add_column("ID", style="dim cyan", min_width=4, justify="right")
            table.add_column("Filename", style="yellow", min_width=15, ratio=2, no_wrap=False)
            table.add_column("SHA-256", min_width=15, ratio=3, no_wrap=False)
            table.add_column("Size", justify="right", min_width=10)
            table.add_column("Mode", justify="center", min_width=9)
            table.add_column("Parts", justify="right", min_width=7)
            table.add_column("Location", style="dim blue", min_width=20, ratio=3, no_wrap=False)
            table.add_column("Timestamp (UTC)", style="dim", min_width=26)
            if PYTZ_AVAILABLE and LOCAL_TZ:
                local_tz_name = LOCAL_TIMEZONE_STR.split('/')[-1].replace('_', ' ')
                table.add_column(f"Timestamp ({local_tz_name})", style="dim", min_width=26)
            else:
                table.add_column("Timestamp (Local)", style="dim", min_width=22)
            if is_trash_list:
                table.add_column("Deleted At (UTC)", style="red", min_width=26) # Specific column for trash

            dt_format_12hr = '%Y-%m-%d %I:%M:%S %p %Z'; dt_format_24hr = '(24hr: %H:%M:%S)'

            for entry in entries:
                entry_id = entry.get('id', -1);
                if entry_id != -1 : valid_ids.add(entry_id) # Only add valid IDs
                size_str = format_bytes(entry.get('file_size_bytes', 0))
                mode_raw = entry.get('mode', 'unknown')
                mode_color = "[green]" if mode_raw == 'process' else "[magenta]"
                mode_str = f"{mode_color}{mode_raw.upper()}[/]"
                sha_str = entry.get('original_sha256', 'N/A')
                parts_str = str(entry.get('number_of_parts', '?'))
                filename_str = entry.get('original_filename', 'N/A')

                # --- CORRECTED Location String Logic ---
                repo = entry.get('github_repo'); tag = entry.get('github_release_tag'); loc_dir = entry.get('output_directory')
                if repo and tag:
                    loc_str = f"[bold]GH:[/bold] {repo} @ {tag}"
                elif loc_dir:
                    try:
                        # Try to get the base name (the last part of the path)
                        base_name = os.path.basename(loc_dir)
                        # If basename is found and not empty, use it prefixed with ".../"
                        # Otherwise, just use the full directory path
                        display_path = f"...{os.sep}{base_name}" if base_name else loc_dir
                        loc_str = f"[bold]Local:[/bold] {display_path}"
                    except Exception:
                        # Fallback if os.path.basename fails for any reason
                        loc_str = f"[bold]Local:[/bold] {loc_dir}"
                else:
                    loc_str = "N/A"
                # --- END CORRECTED Location String Logic ---

                utc_dt_str_iso = entry.get('processing_datetime_utc', ''); local_dt_stacked = "N/A"; utc_dt_stacked = "N/A"
                deleted_utc_stacked = "N/A" # For trash table

                # Format processing time
                if utc_dt_str_iso:
                     try:
                         iso_str = utc_dt_str_iso.replace('Z', '+00:00'); utc_dt = datetime.datetime.fromisoformat(iso_str)
                         if utc_dt.tzinfo is None: utc_dt = pytz.utc.localize(utc_dt)
                         utc_12 = utc_dt.strftime(dt_format_12hr); utc_24 = utc_dt.strftime(dt_format_24hr)
                         utc_dt_stacked = f"{utc_12}\n{utc_24}"
                         if PYTZ_AVAILABLE and LOCAL_TZ:
                              local_dt = utc_dt.astimezone(LOCAL_TZ); local_12 = local_dt.strftime(dt_format_12hr); local_24 = local_dt.strftime(dt_format_24hr)
                              local_dt_stacked = f"{local_12}\n{local_24}"
                         else: local_dt_stacked = "Requires pytz"
                     except Exception as dt_err: print_log("WARN", f"Date parse error '{utc_dt_str_iso}': {dt_err}"); utc_dt_stacked = utc_dt_str_iso; local_dt_stacked = "Error"

                # Format deletion time (if trash list)
                if is_trash_list:
                    deleted_dt_str_iso = entry.get('deleted_datetime_utc', '')
                    if deleted_dt_str_iso:
                         try:
                             iso_str = deleted_dt_str_iso.replace('Z', '+00:00'); del_utc_dt = datetime.datetime.fromisoformat(iso_str)
                             if del_utc_dt.tzinfo is None: del_utc_dt = pytz.utc.localize(del_utc_dt)
                             del_utc_12 = del_utc_dt.strftime(dt_format_12hr); del_utc_24 = del_utc_dt.strftime(dt_format_24hr)
                             deleted_utc_stacked = f"{del_utc_12}\n{del_utc_24}"
                         except Exception as dt_err: print_log("WARN", f"Deleted date parse error '{deleted_dt_str_iso}': {dt_err}"); deleted_utc_stacked = deleted_dt_str_iso

                row_data = [str(entry_id), filename_str, sha_str, size_str, mode_str, parts_str, loc_str, utc_dt_stacked]
                row_data.append(local_dt_stacked)
                if is_trash_list: row_data.append(deleted_utc_stacked) # Add delete time to row
                table.add_row(*row_data)
            console.print(table)

        except Exception as rich_err:
             print_log("ERROR", f"Failed to render table using 'rich': {rich_err}"); use_rich = False; valid_ids.clear()

    if not use_rich: # Fallback basic print
        print("-" * 120); print(f"--- {title} ---")
        for entry in entries:
             entry_id = entry.get('id', -1);
             if entry_id != -1: valid_ids.add(entry_id) # Only add valid IDs
             size_str = format_bytes(entry.get('file_size_bytes', 0))
             mode_raw = entry.get('mode', 'unknown'); mode_color = Fore.GREEN if mode_raw == 'process' else Fore.MAGENTA; mode_str = f"{mode_color}{mode_raw.upper()}{Style.RESET_ALL}"
             sha_raw = entry.get('original_sha256', 'N/A'); parts_print = entry.get('number_of_parts', '?'); filename_print = entry.get('original_filename', 'N/A')

             # --- CORRECTED Location String Logic (Fallback Print) ---
             repo = entry.get('github_repo'); tag = entry.get('github_release_tag'); loc_dir = entry.get('output_directory')
             if repo and tag:
                 loc_print = f"GH: {repo} @ {tag}"
             elif loc_dir:
                 try:
                     base_name = os.path.basename(loc_dir)
                     display_path = f"...{os.sep}{base_name}" if base_name else loc_dir
                     loc_print = f"Local: {display_path}"
                 except Exception:
                     loc_print = f"Local: {loc_dir}" # Fallback
             else:
                 loc_print = "N/A"
             # --- END CORRECTED Location String Logic ---

             utc_dt_str_iso = entry.get('processing_datetime_utc', 'N/A'); local_dt_print = "N/A (Requires pytz)"; utc_dt_print = utc_dt_str_iso
             deleted_dt_print = "N/A" # For trash

             if PYTZ_AVAILABLE and utc_dt_str_iso != 'N/A':
                  try:
                      iso_str = utc_dt_str_iso.replace('Z', '+00:00'); utc_dt = datetime.datetime.fromisoformat(iso_str)
                      if utc_dt.tzinfo is None: utc_dt = pytz.utc.localize(utc_dt)
                      utc_dt_print = utc_dt.strftime('%Y-%m-%d %H:%M:%S %Z')
                      if LOCAL_TZ: local_dt = utc_dt.astimezone(LOCAL_TZ); local_dt_print = local_dt.strftime('%Y-%m-%d %H:%M:%S %Z')
                  except Exception: utc_dt_print = utc_dt_str_iso; local_dt_print = "Error"

             if is_trash_list:
                 deleted_dt_str_iso = entry.get('deleted_datetime_utc', 'N/A')
                 if deleted_dt_str_iso != 'N/A':
                     try:
                         iso_str = deleted_dt_str_iso.replace('Z', '+00:00'); del_utc_dt = datetime.datetime.fromisoformat(iso_str)
                         if del_utc_dt.tzinfo is None: del_utc_dt = pytz.utc.localize(del_utc_dt)
                         deleted_dt_print = del_utc_dt.strftime('%Y-%m-%d %H:%M:%S %Z')
                     except Exception: deleted_dt_print = deleted_dt_str_iso

             print(f"  {Fore.CYAN}{entry_id:>3}:{Style.RESET_ALL} {Fore.YELLOW}{filename_print}{Style.RESET_ALL}")
             print(f"      Mode: {mode_str} | Parts: {parts_print} | Size: {size_str}")
             print(f"      SHA: {sha_raw}")
             print(f"      Loc: {Fore.BLUE}{loc_print}{Style.RESET_ALL}")
             print(f"      Processed (UTC): {utc_dt_print}")
             if not (PYTZ_AVAILABLE and LOCAL_TZ): print(f"      Processed (Local): {local_dt_print}") # Show pytz warning if needed
             else: print(f"      Processed ({LOCAL_TIMEZONE_STR}): {local_dt_print}")
             if is_trash_list: print(f"      {Fore.RED}Deleted At (UTC):  {deleted_dt_print}{Style.RESET_ALL}")
             print("-" * 60)
        print("--- End of List ---")

    return valid_ids # Return the set of displayed IDs

def decrypt_main(args: argparse.Namespace):
    """Handles file reconstruction/decryption using database entries via server."""
    mode_display_name = "RECONSTRUCTION / DECRYPTION"
    start_process_time = time.monotonic()
    print_log("STEP", f"Initiating {mode_display_name} using Database Server: {DB_SERVER_URL}")
    console = Console() # Instantiate console

    if not RICH_AVAILABLE: print_log("WARN", "'rich' library not found. Table formatting/prompts basic.")
    if not PYTZ_AVAILABLE: print_log("WARN", "'pytz' library not found. Local times cannot be displayed.")

    # 1. Query Database (via Server) for ACTIVE entries
    available_entries = db_query_entries(query_trash=False) # <-- Use updated function
    if available_entries is None: print_log("ERROR", "Failed to retrieve entries from database server."); sys.exit(1)

    # --- MODIFIED: Display Entries using helper ---
    print_log("INFO", "Available active entries:")
    valid_ids = _display_entries_table(available_entries, console, is_trash_list=False)
    if not valid_ids: sys.exit(0) # Exit if no entries were displayed

    # 2. Get Choice
    selected_id = None
    while selected_id is None:
        try:
            # --- MODIFIED: Use prompt helper ---
            choice = prompt_user_input(console, "Enter ID to reconstruct/decrypt (or 'q' to quit): ").strip()
            if choice.lower() == 'q': print_log("INFO", "Operation cancelled."); sys.exit(0)
            selected_id_int = int(choice)
            if selected_id_int in valid_ids: selected_id = selected_id_int
            else: print_log("WARN", f"Invalid ID '{selected_id_int}'. Choose from the list.")
        except ValueError: print_log("WARN", "Invalid input. Enter a number or 'q'.")
        except (EOFError, KeyboardInterrupt): print_log("WARN", "\nOperation cancelled."); sys.exit(130)

    # 3. Fetch Details (via Server) for the ACTIVE entry
    print_log("INFO", f"Fetching full details for selected ID: {selected_id}")
    db_details = db_get_entry_details(selected_id) # Fetches active entry
    if not db_details or 'metadata' not in db_details: print_log("ERROR", f"Failed retrieving/parsing details for ID {selected_id} from server."); sys.exit(1)

    metadata = db_details.get('metadata', {})
    original_output_dir = db_details.get('output_directory') # Original local path where parts were saved
    github_repo = db_details.get('github_repo')
    github_tag = db_details.get('github_release_tag')
    reconstruction_mode = db_details.get('mode')
    original_filename = db_details.get('original_filename', 'output.file')
    original_sha256 = db_details.get('original_sha256')
    num_parts = metadata.get('number_of_parts', -1)

    if not reconstruction_mode or not original_sha256 or num_parts == -1: print_log("ERROR", f"Incomplete metadata for ID {selected_id}."); sys.exit(1)
    if reconstruction_mode not in ['process', 'encrypt']: print_log("ERROR", f"Unknown mode '{reconstruction_mode}' for ID {selected_id}."); sys.exit(1)
    print_log("INFO", f"Selected: ID={selected_id}, File='{original_filename}', Mode={reconstruction_mode.upper()}")
    if reconstruction_mode == 'encrypt' and not CRYPTOGRAPHY_AVAILABLE: print_log("ERROR", "'cryptography' required for 'encrypt' mode."); sys.exit(1)

    # 4. Determine Source Directory (Try GitHub download first)
    source_directory = None; attempted_download = False; temp_download_dir_path = None
    hash_buffer_size = args.buffer_size * 1024 * 1024;

    if github_repo and github_tag:
        print_log("INFO", f"Entry has GitHub info ({github_repo} @ {github_tag}).")
        try:
            # --- MODIFIED: Use prompt helper ---
            if prompt_user_confirm(console, "Attempt pull from GitHub first? ", default_val=True):
                attempted_download = True
                print_log("INFO", "Attempting to download layers from GitHub...")
                temp_download_dir_path = download_parts_from_github(repo_full_name=github_repo, release_tag=github_tag,
                    metadata=metadata, download_workers=args.download_workers, hash_buffer_size=hash_buffer_size)
                if temp_download_dir_path and os.path.isdir(temp_download_dir_path):
                    source_directory = temp_download_dir_path
                    print_log("SUCCESS", f"GitHub pull successful. Using temp directory: {temp_download_dir_path}")
                else: print_log("WARN", "GitHub pull failed, cancelled, or invalid dir."); temp_download_dir_path = None
            else: print_log("INFO", "Skipping GitHub pull.")
        except (EOFError, KeyboardInterrupt):
             print_log("WARN", "\nOperation cancelled during prompt.");
             if temp_download_dir_path and os.path.exists(temp_download_dir_path): 
                try: 
                    shutil.rmtree(temp_download_dir_path) 
                except OSError: 
                    pass
             sys.exit(130)

    if not source_directory:
        if not original_output_dir:
             print_log("ERROR", "No GitHub source used and no original local directory stored in DB.")
             if attempted_download: print_log("ERROR", "Cannot proceed without source layers.")
             sys.exit(1)
        print_log("INFO", f"Using original local directory from DB: '{original_output_dir}'")
        if not os.path.isdir(original_output_dir): print_log("ERROR", f"Original directory '{original_output_dir}' not found."); sys.exit(1)
        source_directory = original_output_dir
    print_log("INFO", f"Using source directory for layers: '{source_directory}'")

    # 5. Setup Output File
    output_file_path = None
    try:
        output_file_path_arg = args.output_file; default_output_path = os.path.join(os.getcwd(), original_filename)
        output_file_path = os.path.abspath(output_file_path_arg or default_output_path)
        output_parent_dir = os.path.dirname(output_file_path)
        if output_parent_dir and not os.path.exists(output_parent_dir): print_log("INFO", f"Creating output directory: '{output_parent_dir}'"); os.makedirs(output_parent_dir, exist_ok=True)
        print_log("INFO", f"Output file will be: '{output_file_path}'")
        if os.path.exists(output_file_path):
            # --- MODIFIED: Use prompt helper ---
            if not prompt_user_confirm(console, f"Output file '{os.path.basename(output_file_path)}' exists. Overwrite? ", default_val=False):
                print_log("INFO", "Operation cancelled to avoid overwrite.");
                if temp_download_dir_path and os.path.exists(temp_download_dir_path): 
                    try: 
                        shutil.rmtree(temp_download_dir_path) 
                    except OSError: 
                        pass
                sys.exit(0)
            print_log("WARN", f"Overwriting existing file: '{output_file_path}'")
            try: os.remove(output_file_path)
            except OSError as rm_err: print_log("ERROR", f"Failed to remove existing file: {rm_err}"); sys.exit(1)
    except (OSError, EOFError, KeyboardInterrupt) as e:
        print_log("ERROR", f"Failed setting up output file: {e}")
        if isinstance(e, (EOFError, KeyboardInterrupt)): print_log("INFO", "Cancelled.")
        if temp_download_dir_path and os.path.exists(temp_download_dir_path): 
            try: 
                shutil.rmtree(temp_download_dir_path) 
            except OSError: 
                pass
        sys.exit(1 if isinstance(e, OSError) else 130)

    # 6. Perform Reconstruction / Decryption
    reconstruction_ok = False; final_sha256: Optional[str] = None; final_file_size: int = 0
    max_workers = args.workers; expected_size = metadata.get('file_size_bytes')
    if expected_size is None: print_log("ERROR", "File size missing in metadata."); sys.exit(1)
    final_exit_code = 1 # Default error

    if reconstruction_mode == 'encrypt':
        if num_parts == 0:
            print_log("INFO", "Reconstructing empty encrypted file (0 parts).");
            try:
                with open(output_file_path, 'wb') as f: pass; empty_sha = hashlib.sha256(b'').hexdigest()
                if os.path.getsize(output_file_path) == 0 and empty_sha == original_sha256: print_log("SUCCESS", "Empty file ok."); reconstruction_ok = True; final_file_size = 0; final_sha256 = empty_sha
                else: print_log("ERROR", f"Empty file SHA mismatch! Exp {original_sha256[:8]}.., Got {empty_sha[:8]}..."); reconstruction_ok = False
            except Exception as e: print_log("ERROR", f"Failed creating empty output: {e}"); reconstruction_ok = False
        else: # num_parts > 0
            print_log("STEP", f"Starting parallel DECRYPTION / EXTRACTION ({max_workers} workers)")
            tasks = []; all_parts_ok = True; layer_status_decrypt: Dict[str, str] = {}; status_lock_decrypt = threading.Lock()
            try:
                print_log("INFO", "Pre-checking layers and keys...")
                if not isinstance(metadata.get("parts"), list): raise ValueError("Metadata 'parts' missing/invalid.")
                for part_meta in metadata["parts"]:
                    part_num = part_meta.get("part_number"); enc_sha = part_meta.get("encrypted_sha256"); key_b64 = part_meta.get("decryption_key_base64")
                    if part_num is None or enc_sha is None or key_b64 is None: print_log("ERROR", f"Incomplete metadata part {part_num}."); all_parts_ok = False; break
                    part_filename = part_meta.get("part_filename") or f"part_{part_num}" # Use stored name
                    part_filepath = os.path.join(source_directory, part_filename)
                    if not os.path.isfile(part_filepath): print_log("ERROR", f"Layer file missing: '{part_filename}'"); all_parts_ok = False; break
                    try:
                        key_bytes = base64.b64decode(key_b64, validate=True)
                        if len(key_bytes) != 32: raise ValueError(f"Invalid key length ({len(key_bytes)}) for part {part_num}.")
                        tasks.append({'part_number': part_num, 'part_filepath': part_filepath, 'expected_sha256': enc_sha,
                                      'decryption_key': key_bytes, 'hash_buffer_size': hash_buffer_size,
                                      'layer_status': layer_status_decrypt, 'status_lock': status_lock_decrypt })
                        layer_status_decrypt[part_filename] = LAYER_WAITING
                    except (TypeError, base64.binascii.Error, ValueError) as key_e: print_log("ERROR", f"Invalid key layer {part_num}: {key_e}"); all_parts_ok = False; break
                if not all_parts_ok: raise ValueError("Missing layers or invalid keys during pre-check.")
                if len(tasks) != num_parts: raise ValueError(f"Metadata part count ({num_parts}) != tasks found ({len(tasks)}).")
                tasks.sort(key=lambda t: t['part_number'])
                print_log("INFO", "Layers/keys valid. Starting extraction & assembly.")
            except ValueError as e: print_log("ERROR", f"Pre-check failed: {e}. Aborting."); reconstruction_ok = False; all_parts_ok = False

            if all_parts_ok:
                next_part_to_write = 1; decrypted_part_buffer: Dict[int, bytes] = {}; assembly_failed = False; total_bytes_written_assembly = 0; assembly_lock = threading.Lock(); start_assembly_time = time.monotonic()
                try:
                    print_log("STEP", "Extracting layers and combining...")
                    with open(output_file_path, 'wb') as outfile, ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='DecWorker') as executor:
                        futures: List[Future] = [executor.submit(process_decryption_part, task) for task in tasks]
                        print_log("INFO", f"Submitted {len(tasks)} extraction tasks. Assembling...")
                        pbar = tqdm(total=len(futures), unit='layer', desc=f"{Fore.YELLOW}Extract & Combine{Style.RESET_ALL}", bar_format=TQDM_BAR_FORMAT, ncols=90, leave=False)
                        try:
                            for future in as_completed(futures):
                                if assembly_failed: continue
                                try:
                                    part_num, decrypted_data, success = future.result(); pbar.update(1)
                                    if not success or decrypted_data is None: assembly_failed = True; print_log("ERROR", f"Stopping assembly: extraction failure layer {part_num}."); continue
                                    with assembly_lock:
                                        if part_num == next_part_to_write:
                                            try: 
                                                outfile.write(decrypted_data)
                                                total_bytes_written_assembly += len(decrypted_data)
                                                next_part_to_write += 1
                                                del decrypted_data
                                                while next_part_to_write in decrypted_part_buffer: 
                                                    buffered_data = decrypted_part_buffer.pop(next_part_to_write) 
                                                    outfile.write(buffered_data); total_bytes_written_assembly += len(buffered_data)
                                                    next_part_to_write += 1
                                                    del buffered_data
                                            except IOError as write_e: print_log("ERROR", f"IOError writing part {part_num}: {write_e}"); assembly_failed = True
                                        elif part_num > next_part_to_write: decrypted_part_buffer[part_num] = decrypted_data
                                        else: print_log("WARN", f"Received already written part {part_num}. Discarding."); del decrypted_data
                                except Exception as worker_e: print_log("ERROR", f"Exception processing extraction result: {worker_e}"); traceback.print_exc(); assembly_failed = True
                        except KeyboardInterrupt: 
                            print_log("WARN", "\nExtraction/Assembly cancelled.") 
                            assembly_failed = True; print_log("WARN", "Cancelling tasks...")
                            for f in futures: 
                                f.cancel()
                                time.sleep(0.5)
                        finally: 
                            if pbar: 
                                pbar.close()
                    if not assembly_failed:
                        if next_part_to_write != num_parts + 1: raise RuntimeError(f"Assembly incomplete. Wrote up to part {next_part_to_write - 1}.")
                        if decrypted_part_buffer: raise RuntimeError(f"Assembly finished, but buffer has parts: {list(decrypted_part_buffer.keys())}.")
                        end_assembly_time = time.monotonic()
                        print_log("SUCCESS", f"Extraction & assembly completed in {format_duration(end_assembly_time - start_assembly_time)}.")
                        reconstruction_ok = True
                except (RuntimeError, ValueError, OSError, Exception) as e: 
                    print_log("ERROR", f"Extraction/Assembly process failed: {e}")
                    if not isinstance(e, (RuntimeError, ValueError, OSError, KeyboardInterrupt)): 
                        traceback.print_exc() 
                        assembly_failed = True
                if assembly_failed: reconstruction_ok = False

    elif reconstruction_mode == 'process':
        print_log("STEP", f"Starting plaintext file RECONSTRUCTION from: {source_directory}")
        reconstruction_ok = reconstruct_plaintext_file(metadata=metadata, source_directory=source_directory, output_file_path=output_file_path, hash_buffer_size=hash_buffer_size)
        if not reconstruction_ok: print_log("ERROR", "Plaintext reconstruction failed.")
    else: print_log("ERROR", f"Internal Error: Unknown mode '{reconstruction_mode}'"); reconstruction_ok = False

    # 7. Final Verification & Summary
    if not reconstruction_ok:
        print_log("ERROR", f"{mode_display_name} failed or cancelled before completion.")
        if os.path.exists(output_file_path) and os.path.getsize(output_file_path) > 0 :
             print_log("WARN", f"Output file '{output_file_path}' may be incomplete/corrupted.")
             try: # Use helper for confirm prompt
                 # --- MODIFIED: Use prompt helper for cleanup confirmation ---
                 if prompt_user_confirm(console, "Delete incomplete/corrupted output file? ", default_val=False):
                     os.remove(output_file_path); print_log("INFO", "Deleted incomplete output file.")
                 else: print_log("INFO", "Incomplete output file kept.")
             except (OSError, EOFError, KeyboardInterrupt) as clean_e: print_log("WARN", f"Cleanup of incomplete output failed/skipped: {clean_e}")
        final_exit_code = 1
    else: # Reconstruction OK, verify
        print_log("STEP", "Verifying final file integrity (Size and SHA-256)...")
        verification_passed = False
        try:
            if not os.path.isfile(output_file_path): raise FileNotFoundError(f"CRITICAL: Output file missing after reconstruction!")
            final_file_size = os.path.getsize(output_file_path)
            print_log("INFO", f"Reconstructed size: {format_bytes(final_file_size)}.")
            if final_file_size != expected_size: raise ValueError(f"Final size check FAILED! Disk: {final_file_size:,}, Expected: {expected_size:,}.")
            else: print_log("INFO", "File size matches.")
            final_sha256 = calculate_sha256(output_file_path, final_file_size, hash_buffer_size, "Verify Final SHA", leave=True)
            if not final_sha256: raise Exception("Final SHA-256 calculation failed.")
            print_log("INFO", f"Reconstructed SHA256: {Fore.YELLOW}{final_sha256}{Style.RESET_ALL}")
            print_log("INFO", f"Expected SHA256:      {Fore.GREEN}{original_sha256}{Style.RESET_ALL}")
            if final_sha256 == original_sha256: print_log("SUCCESS", "Checksum PASSED! File reconstructed successfully."); verification_passed = True; final_exit_code = 0
            else: raise ValueError("Checksum FAILED! Reconstructed file content mismatch.")
        except (FileNotFoundError, ValueError, Exception) as verify_e: print_log("ERROR", f"Final verification failed: {verify_e}"); verification_passed = False; final_exit_code = 1

        # --- Print Final Summary ---
        end_process_time = time.monotonic(); total_time = end_process_time - start_process_time
        summary_status = "SUCCESS" if verification_passed else "ERROR"; status_color = Fore.GREEN if verification_passed else Fore.RED
        print(f"\n{status_color}{Style.BRIGHT}--- {mode_display_name} Complete ---{Style.RESET_ALL}")
        print(f"  {'DB Entry ID:':<25} {selected_id}")
        source_label = f"{' (Temporary)' if source_directory == temp_download_dir_path else ''}"
        print(f"  {'Source layers from:':<25} '{source_directory}'{source_label}")
        print(f"  {'Reconstructed file:':<25} '{output_file_path}'")
        if 'final_file_size' in locals(): print(f"  {'Reconstructed Size:':<25} {format_bytes(final_file_size)} ({final_file_size:,} bytes)")
        status_text = f"({Fore.GREEN}Match{Style.RESET_ALL})" if verification_passed else f"({Fore.RED}MISMATCH{Style.RESET_ALL})"
        print(f"  {'Verified SHA-256:':<25} {final_sha256 if final_sha256 else Fore.RED+'Verification Not Completed'+Style.RESET_ALL} {status_text if final_sha256 else ''}")
        print(f"  {'Total layers processed:':<25} {num_parts}")
        if reconstruction_mode == 'encrypt': print(f"  {'Extraction Workers:':<25} {max_workers}")
        if attempted_download: pull_workers_used = min(args.download_workers, MAX_DOWNLOAD_WORKERS, num_parts if num_parts > 0 else 1); print(f"  {'Pull Workers Used:':<25} {pull_workers_used}")
        print(f"  {'Total time taken:':<25} {format_duration(total_time)}")
        print(f"{status_color}{Style.BRIGHT}-----------------------------------{Style.RESET_ALL}\n")

    # 8. Cleanup Temp Dir
    if temp_download_dir_path and os.path.exists(temp_download_dir_path) and os.path.isdir(temp_download_dir_path):
        try:
            # --- MODIFIED: Use prompt helper ---
            if prompt_user_confirm(console, f"Delete temporary download directory '{os.path.basename(temp_download_dir_path)}'? ", default_val=False):
                print_log("INFO", f"Removing temporary download directory: {temp_download_dir_path}")
                shutil.rmtree(temp_download_dir_path); print_log("INFO", "Temporary directory removed.")
            else: print_log("INFO", "Temporary download directory kept.")
        except (OSError, EOFError, KeyboardInterrupt) as clean_e: print_log("WARN", f"Failed/skipped temp dir cleanup: {clean_e}")
        except Exception as final_clean_e: print_log("WARN", f"Unexpected error during final temp cleanup: {final_clean_e}")

    # 9. Exit
    sys.exit(final_exit_code)
# --- End of decrypt_main function ---
# --- NEW 'list' Main Function ---

def list_main(args: argparse.Namespace):
    """Handles listing active or trashed entries."""
    is_trash = args.trash
    mode_display_name = "Trash Listing" if is_trash else "Active Entry Listing"
    start_process_time = time.monotonic()
    print_log("STEP", f"Initiating {mode_display_name} via DB Server: {DB_SERVER_URL}")
    console = Console()

    if not RICH_AVAILABLE: print_log("WARN", "'rich' library not found. Table formatting basic.")
    if not PYTZ_AVAILABLE: print_log("WARN", "'pytz' library not found. Local times cannot be displayed.")

    # 1. Query Database (via Server)
    entries = db_query_entries(query_trash=is_trash) # Use updated function
    if entries is None:
        print_log("ERROR", f"Failed to retrieve {'trash' if is_trash else 'active'} entries from database server.")
        sys.exit(1)

    # 2. Display Entries using helper
    _display_entries_table(entries, console, is_trash_list=is_trash)

    # 3. Final Summary
    end_process_time = time.monotonic()
    total_time = end_process_time - start_process_time
    print_log("SUCCESS", f"{mode_display_name} complete. Time: {format_duration(total_time)}")
    sys.exit(0)


# --- NEW 'delete' Main Function ---
# --- UPDATED 'delete' Main Function with Wait-and-Check Logic ---

def delete_main(args: argparse.Namespace):
    """Handles deleting an entry (moves to trash, deletes GitHub assets AND release),
       using a wait-and-check method to confirm release deletion."""
    mode_display_name = "ENTRY DELETION"
    start_process_time = time.monotonic()
    print_log("STEP", f"Initiating {mode_display_name} using Database Server: {DB_SERVER_URL}")
    console = Console() # Assumes RICH_AVAILABLE check happened earlier

    if not PYGITHUB_AVAILABLE:
        print_log("WARN", "'PyGithub' not found. Cannot delete GitHub assets/releases if applicable.")

    # --- Steps 1-5: Querying, Displaying, Getting Choice, Fetching Details, Confirmation (Identical to previous version) ---
    # 1. Query *Active* Entries
    available_entries = db_query_entries(query_trash=False)
    if available_entries is None:
        print_log("ERROR", "Failed to retrieve active entries from database server.")
        sys.exit(1)

    # 2. Display Active Entries
    print_log("INFO", "Available active entries to delete:")
    valid_ids = _display_entries_table(available_entries, console, is_trash_list=False)
    if not valid_ids:
        sys.exit(0) # Exit if no entries to delete

    # 3. Get Choice
    selected_id = None
    while selected_id is None:
        try:
            choice = prompt_user_input(console, "Enter ID of the entry to delete (or 'q' to quit): ").strip()
            if choice.lower() == 'q':
                print_log("INFO", "Operation cancelled.")
                sys.exit(0)
            selected_id_int = int(choice)
            if selected_id_int in valid_ids:
                selected_id = selected_id_int
            else:
                print_log("WARN", f"Invalid ID '{selected_id_int}'. Choose from the list.")
        except ValueError:
            print_log("WARN", "Invalid input. Enter a number or 'q'.")
        except (EOFError, KeyboardInterrupt):
            print_log("WARN", "\nOperation cancelled.")
            sys.exit(130)

    # 4. Fetch Details
    print_log("INFO", f"Fetching details for entry ID {selected_id} to proceed with deletion...")
    db_details = db_get_entry_details(selected_id) # Fetches active entry
    if not db_details or 'metadata' not in db_details:
        print_log("ERROR", f"Failed retrieving/parsing details for ID {selected_id}. Cannot proceed.")
        sys.exit(1)

    metadata = db_details.get('metadata', {})
    original_filename = db_details.get('original_filename', 'N/A')
    original_sha256 = db_details.get('original_sha256', 'N/A')
    github_repo = db_details.get('github_repo')
    github_tag = db_details.get('github_release_tag')
    entry_mode = db_details.get('mode')

    print_log("INFO", f"Selected for deletion: ID={selected_id}, File='{original_filename}', SHA={original_sha256[:8]}..., Mode={entry_mode}")

    # 5. Confirmation
    print_log("WARN", f"{Style.BRIGHT}{Fore.RED}--- WARNING ---{Style.RESET_ALL}")
    print_log("WARN", "This operation will:")
    print_log("WARN", f"  1. Move database entry ID {selected_id} ('{original_filename}') to the {Style.BRIGHT}'trash'{Style.RESET_ALL} table.")
    if github_repo and github_tag:
        print_log("WARN", f"  2. {Fore.RED}{Style.BRIGHT}PERMANENTLY DELETE{Style.RESET_ALL}{Fore.RED} associated file assets from GitHub release '{github_tag}'.{Style.RESET_ALL}")
        print_log("WARN", f"  3. {Fore.RED}{Style.BRIGHT}PERMANENTLY DELETE{Style.RESET_ALL}{Fore.RED} the GitHub release '{github_tag}' itself in repo '{github_repo}'.{Style.RESET_ALL}")
    else:
        print_log("WARN", f"  2. {Fore.YELLOW}No GitHub release info found; assets/release will NOT be deleted from GitHub.{Style.RESET_ALL}")
    print_log("WARN", f"{Style.BRIGHT}{Fore.RED}GitHub asset and release deletion CANNOT be undone.{Style.RESET_ALL}")

    try:
        if not prompt_user_confirm(console, f"Are you absolutely sure you want to delete entry {selected_id}?", default_val=False):
            print_log("INFO", "Deletion cancelled by user.")
            sys.exit(0)

        if github_repo and github_tag:
            confirm_gh_prompt = (f"Type '{Fore.RED}yes{Style.RESET_ALL}' to confirm "
                                 f"{Fore.RED}{Style.BRIGHT}PERMANENT DELETION{Style.RESET_ALL}{Fore.RED} of GitHub assets AND the release "
                                 f"for ID {selected_id}:{Style.RESET_ALL} ")
            gh_confirm_response = prompt_user_input(console, confirm_gh_prompt).strip().lower()
            if gh_confirm_response != 'yes':
                print_log("INFO", "GitHub asset/release deletion not confirmed. Aborting operation.")
                sys.exit(0)
            print_log("WARN", "GitHub asset and release deletion confirmed by user.")

    except (EOFError, KeyboardInterrupt):
        print_log("WARN", "\nOperation cancelled during confirmation.")
        sys.exit(130)
    # --- End of Confirmation ---

    # --- Step 6a: Perform GitHub Asset Deletion (Identical logic to previous versions) ---
    github_asset_deletion_successful = True
    github_assets_deleted = 0
    github_assets_failed = 0
    github_assets_skipped = 0
    release_object_for_deletion: Optional[GitRelease] = None
    gh = None # Initialize gh here

    if github_repo and github_tag and PYGITHUB_AVAILABLE:
        print_log("STEP", f"Deleting GitHub Release Assets for Tag: {github_tag} in Repo: {github_repo}")
        repo = None # Initialize repo here
        try:
            _, github_token = get_github_credentials(prompt_for_repo=False)
            if not github_token: raise ValueError("GitHub token required for asset deletion.")
            gh = _get_github_instance(github_token) # gh is assigned here
            if not gh: raise ConnectionError("Failed GitHub authentication for deletion.")
            repo = _get_repo(gh, github_repo) # repo is assigned here
            if not repo: raise ValueError(f"Repository '{github_repo}' not found/inaccessible.")

            print_log("INFO", f"Fetching release '{github_tag}'...")
            try:
                release_object_for_deletion = repo.get_release(github_tag)
            except UnknownObjectException:
                print_log("WARN", f"Release '{github_tag}' not found before asset deletion (maybe already deleted?).")
                release_object_for_deletion = None

            if release_object_for_deletion:
                print_log("INFO", f"Found release '{release_object_for_deletion.title}'. Fetching assets...")
                # ... (rest of asset fetching and deletion logic is identical) ...
                try: assets_on_release = list(release_object_for_deletion.get_assets())
                except Exception as e: print_log("ERROR", f"Could not list assets for release '{github_tag}': {e}"); assets_on_release = []

                expected_asset_names = set()
                parts_meta = metadata.get("parts", [])
                if isinstance(parts_meta, list):
                    for p in parts_meta:
                        if isinstance(p, dict):
                             fname = p.get("part_filename") or f"part_{p.get('part_number')}"
                             if fname: expected_asset_names.add(fname)
                        else: print_log("WARN", f"Skipping invalid part metadata item: {p}")
                else: print_log("WARN", f"Metadata 'parts' key is not a list or is missing.")

                if entry_mode == 'process' and 'original_sha256' in metadata and metadata.get('original_sha256'):
                    json_asset_name = f"{metadata['original_sha256']}.json"
                    expected_asset_names.add(json_asset_name)

                print_log("INFO", f"Expecting {len(expected_asset_names)} assets: {', '.join(sorted(list(expected_asset_names)))}")
                print_log("INFO", f"Found {len(assets_on_release)} assets on release.")

                if not assets_on_release:
                    print_log("INFO", "No assets found on the release to delete.")
                else:
                    for asset in assets_on_release:
                        part_num_str = asset.name.split('_')[-1] if asset.name.startswith('part_') else None
                        if asset.name in expected_asset_names:
                            if _delete_github_asset(asset, asset.name, part_num_str): github_assets_deleted += 1
                            else: github_assets_failed += 1
                        else:
                            print_log("DEBUG", f"Skipping asset '{asset.name}' (not expected).", part_num=part_num_str)
                            github_assets_skipped += 1

                if github_assets_failed > 0:
                    print_log("ERROR", f"{github_assets_failed} GitHub asset deletions failed.")
                    github_asset_deletion_successful = False
                else:
                    print_log("SUCCESS", f"GitHub asset deletion process complete. Del: {github_assets_deleted}, Fail: {github_assets_failed}, Skip: {github_assets_skipped}")
                    github_asset_deletion_successful = True
            # else: release_object_for_deletion was None initially, asset deletion skipped

        except (ValueError, ConnectionError, ConnectionAbortedError, GithubException) as gh_del_err:
            print_log("ERROR", f"GitHub setup/asset deletion failed: {gh_del_err}")
            github_asset_deletion_successful = False
            release_object_for_deletion = None
        except Exception as e:
            print_log("ERROR", f"Unexpected error during GitHub asset deletion: {e}")
            traceback.print_exc()
            github_asset_deletion_successful = False
            release_object_for_deletion = None

    elif github_repo and github_tag and not PYGITHUB_AVAILABLE:
        print_log("ERROR", "PyGithub not available. Skipping GitHub asset/release deletion.")
        github_asset_deletion_successful = False
    # --- End of Step 6a ---


    # --- Step 6b: Delete Release and Confirm with Wait-and-Check ---
    github_release_deletion_confirmed_success = False # Default to False until confirmed
    github_release_deletion_error_details = "Not Applicable" if not (github_repo and github_tag) else "Skipped"
    initial_delete_error = None # Store non-critical error from initial delete attempt
    needs_check = False # Flag to indicate if we need to perform the wait-and-check

    if release_object_for_deletion and github_asset_deletion_successful:
        # Only proceed if release object exists and assets were handled
        print_log("STEP", f"Attempting initial deletion of GitHub release '{github_tag}'...")
        try:
            delete_result = release_object_for_deletion.delete_release()
            if delete_result:
                print_log("INFO", f"Initial delete call for release '{github_tag}' succeeded (returned True). Proceeding to confirmation check.")
                needs_check = True # Confirm it's actually gone
            else:
                print_log("WARN", f"Initial delete call for release '{github_tag}' returned False/None. Proceeding to check.")
                initial_delete_error = "Initial delete call returned non-True"
                needs_check = True

        except UnknownObjectException:
             # Already deleted during the initial call attempt
            print_log("WARN", f"Release '{github_tag}' was already gone during initial delete attempt.")
            github_release_deletion_confirmed_success = True
            github_release_deletion_error_details = "Already deleted"
            needs_check = False # No need to check again
        except RateLimitExceededException as e:
            reset_time_ts = e.headers.get('x-ratelimit-reset', time.time())
            reset_time = datetime.datetime.fromtimestamp(float(reset_time_ts))
            err_msg = f"Rate limit hit during initial delete. Try again after {reset_time.strftime('%Y-%m-%d %H:%M:%S')}."
            print_log("ERROR", f"{err_msg} Error: {e}")
            github_release_deletion_error_details = err_msg
            github_release_deletion_confirmed_success = False
            needs_check = False # Definitive failure
        except GithubException as e:
            # Other GitHub API error during initial delete - log it but still check
            err_msg = f"API Error during initial delete (Status: {e.status}): {e.data.get('message', 'Unknown')}"
            print_log("WARN", f"{err_msg}. Data: {e.data}. Will proceed to check if release is gone.")
            initial_delete_error = err_msg # Store the initial error
            needs_check = True
        except Exception as e:
            # Unexpected error during initial delete - log it but still check
            err_msg = f"Unexpected error during initial delete: {e.__class__.__name__}: {e}"
            print_log("ERROR", f"{err_msg}. Will proceed to check if release is gone.")
            traceback.print_exc()
            initial_delete_error = err_msg # Store the initial error
            needs_check = True

        # --- Wait and Check ---
        if needs_check:
            wait_seconds = 3
            print_log("INFO", f"Waiting {wait_seconds} seconds before checking release status...")
            time.sleep(wait_seconds)
            print_log("INFO", f"Checking if release '{github_tag}' exists after delete attempt...")
            try:
                # We need repo object here. Ensure it's available from Step 6a.
                if repo: # Check if repo object was successfully obtained earlier
                    checked_release = repo.get_release(github_tag)
                    # If get_release succeeds, the release STILL exists!
                    print_log("ERROR", f"Confirmation FAILED: Release '{github_tag}' still exists after delete attempt.")
                    github_release_deletion_error_details = "Still exists after delete attempt"
                    if initial_delete_error:
                         github_release_deletion_error_details += f" (Initial Error: {initial_delete_error})"
                    github_release_deletion_confirmed_success = False
                else:
                    # Should not happen if asset deletion succeeded, but handle defensively
                     print_log("ERROR", "Cannot perform confirmation check: Repository object unavailable.")
                     github_release_deletion_error_details = "Repo object unavailable for check"
                     github_release_deletion_confirmed_success = False

            except UnknownObjectException:
                # THIS IS THE EXPECTED SUCCESS CASE for the check
                print_log("SUCCESS", f"Confirmation PASSED: Release '{github_tag}' no longer exists.")
                github_release_deletion_confirmed_success = True
                github_release_deletion_error_details = None # Clear any temporary initial error
            except RateLimitExceededException as e_check:
                 reset_time_ts = e_check.headers.get('x-ratelimit-reset', time.time())
                 reset_time = datetime.datetime.fromtimestamp(float(reset_time_ts))
                 err_msg = f"Rate limit hit during confirmation check. Try again after {reset_time.strftime('%Y-%m-%d %H:%M:%S')}."
                 print_log("ERROR", f"{err_msg} Error: {e_check}")
                 github_release_deletion_error_details = err_msg
                 github_release_deletion_confirmed_success = False
            except GithubException as e_check:
                err_msg = f"API Error during confirmation check (Status: {e_check.status}): {e_check.data.get('message', 'Unknown')}"
                print_log("ERROR", f"{err_msg}. Data: {e_check.data}")
                github_release_deletion_error_details = err_msg
                github_release_deletion_confirmed_success = False
            except Exception as e_check:
                err_msg = f"Unexpected error during confirmation check: {e_check.__class__.__name__}: {e_check}"
                print_log("ERROR", err_msg)
                traceback.print_exc()
                github_release_deletion_error_details = err_msg
                github_release_deletion_confirmed_success = False

    elif not release_object_for_deletion and github_repo and github_tag:
         # Release wasn't found initially in Step 6a
        github_release_deletion_confirmed_success = True # Already gone is success
        github_release_deletion_error_details = "Not Found Initially"
    elif not github_asset_deletion_successful:
         # Assets failed, so release deletion wasn't attempted
         github_release_deletion_error_details = "Skipped (Asset Deletion Failed)"
         github_release_deletion_confirmed_success = False
    # If no github_repo/tag, error_details remains "Not Applicable" and confirmed_success False (but N/A)

    # --- End of Step 6b ---


    # 7. Move DB Entry to Trash (via Server) - Condition is now based on CONFIRMED success
    db_move_successful = False
    if github_asset_deletion_successful and github_release_deletion_confirmed_success:
        print_log("STEP", f"Moving database entry ID {selected_id} to trash...")
        db_move_successful = db_delete_entry(selected_id)
        if not db_move_successful:
            print_log("ERROR", f"Failed to move database entry ID {selected_id} to trash via server.")
    else:
        reason = []
        if not github_asset_deletion_successful: reason.append("asset deletion failure/skip")
        # Use the confirmed flag here
        if not github_release_deletion_confirmed_success: reason.append("release deletion confirmation failure/skip")
        print_log("WARN", f"Skipping database move to trash due to: {', '.join(reason)}.")


    # 8. Final Summary - Using refined status
    end_process_time = time.monotonic()
    total_time = end_process_time - start_process_time

    # Overall success requires all steps to be confirmed successful or N/A
    overall_success = github_asset_deletion_successful and github_release_deletion_confirmed_success and db_move_successful
    summary_status = "SUCCESS" if overall_success else "ERROR (Partial or Full Failure)"
    status_color = Fore.GREEN if overall_success else Fore.RED

    print(f"\n{status_color}{Style.BRIGHT}--- {mode_display_name} Summary ---{Style.RESET_ALL}")
    print(f"  {'Entry ID Processed:':<27} {selected_id}")
    print(f"  {'File:':<27} '{original_filename}'")

    # GitHub Asset Status Report (No change needed)
    if github_repo and github_tag:
        if github_assets_failed == 0 and github_asset_deletion_successful:
             asset_status = f"{Fore.GREEN}Successful ({github_assets_deleted} deleted)"
        elif github_assets_failed > 0 :
             asset_status = f"{Fore.YELLOW}Partial ({github_assets_deleted} del, {github_assets_failed} fail)"
        else:
            asset_status = f"{Fore.RED}Failed/Skipped{Style.RESET_ALL}"
        print(f"  {'GitHub Asset Deletion:':<27} {asset_status}")

        # GitHub Release Status Report (Refined based on check)
        if github_release_deletion_confirmed_success:
            if github_release_deletion_error_details == "Already deleted":
                 release_status = f"{Fore.YELLOW}Already Deleted{Style.RESET_ALL}"
            elif github_release_deletion_error_details == "Not Found Initially":
                 release_status = f"{Fore.YELLOW}Not Found Initially{Style.RESET_ALL}"
            else: # Confirmed success via check or initial delete attempt worked and was confirmed
                 release_status = f"{Fore.GREEN}Successful (Confirmed Gone){Style.RESET_ALL}"
        else: # Deletion failed or confirmation failed
            error_info = f" ({github_release_deletion_error_details or 'Unknown'})" if github_release_deletion_error_details else ""
            release_status = f"{Fore.RED}Failed/Skipped{error_info}{Style.RESET_ALL}"
        print(f"  {'GitHub Release Deletion:':<27} {release_status}")

    else: # No GitHub repo/tag info
        print(f"  {'GitHub Asset Deletion:':<27} {Fore.CYAN}Not Applicable{Style.RESET_ALL}")
        print(f"  {'GitHub Release Deletion:':<27} {Fore.CYAN}Not Applicable{Style.RESET_ALL}")

    # DB Move Status Report (No change needed)
    db_status = f"{Fore.GREEN}Successful{Style.RESET_ALL}" if db_move_successful else f"{Fore.RED}Failed/Skipped{Style.RESET_ALL}"
    print(f"  {'Database Move to Trash:':<27} {db_status}")

    # Overall Status Report (No change needed)
    print(f"  {'Overall Status:':<27} {status_color}{summary_status}{Style.RESET_ALL}")
    print(f"  {'Total time taken:':<27} {format_duration(total_time)}")
    print(f"{status_color}{Style.BRIGHT}-----------------------------------{Style.RESET_ALL}\n")

    sys.exit(0 if overall_success else 1)

# --- End of delete_main function ---
# --- Argument Parsing Setup ---
def setup_arg_parser():
    """Sets up the command-line argument parser with enhanced, colored help."""
    script_name = os.path.basename(__file__)
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # --- MODIFIED: Updated description ---
        description=f"""
{Style.BRIGHT}{Fore.CYAN}==== File Processor & GitHub Uploader/Downloader ===={Style.RESET_ALL}
Version: {Fore.YELLOW}{PROCESSOR_VERSION}{Style.RESET_ALL}

Splits, encrypts, lists, deletes, and reconstructs files, logging operations via a DB server.
Optionally interacts with GitHub Releases for storing/retrieving/deleting file parts.
""",
        # --- MODIFIED: Updated epilog ---
        epilog=f"""
{Style.BRIGHT}{Fore.MAGENTA}-------------------- MODES --------------------{Style.RESET_ALL}
  {Fore.GREEN}{Style.BRIGHT}process{Style.RESET_ALL}:  Splits file. Logs metadata (+parts SHAs) to DB. Optionally uploads parts+JSON to GH.
  {Fore.MAGENTA}{Style.BRIGHT}encrypt{Style.RESET_ALL}:  Splits & encrypts parts. Logs metadata (+enc SHAs & {Fore.RED}keys{Style.RESET_ALL}) to DB. Optionally uploads {Fore.RED}only parts{Style.RESET_ALL} to GH.
  {Fore.BLUE}{Style.BRIGHT}decrypt{Style.RESET_ALL}:  Reconstructs file from DB entry. Lists active entries, prompts for ID. Optionally pulls layers from GH.
  {Fore.CYAN}{Style.BRIGHT}list{Style.RESET_ALL}:     Lists {Style.BRIGHT}active{Style.RESET_ALL} entries from DB. Use {Fore.YELLOW}--trash{Style.RESET_ALL} to list deleted entries.
  {Fore.RED}{Style.BRIGHT}delete{Style.RESET_ALL}:   Moves DB entry to {Style.BRIGHT}trash{Style.RESET_ALL}. {Fore.RED}{Style.BRIGHT}PERMANENTLY deletes{Style.RESET_ALL}{Fore.RED} associated assets from GitHub Release.{Style.RESET_ALL}

{Style.BRIGHT}{Fore.MAGENTA}--------------- DATABASE SERVER ---------------{Style.RESET_ALL}
* Acts as client to a Flask DB Server ({Fore.GREEN}'{DB_SERVER_SCRIPT}'{Style.RESET_ALL}). Ensure server is running.
* Connects to: {Fore.CYAN}{DB_SERVER_URL}{Style.RESET_ALL}. Attempts to start server if not detected.

{Style.BRIGHT}{Fore.MAGENTA}------------- GITHUB INTEGRATION ------------{Style.RESET_ALL}
* Requires {Fore.CYAN}PyGithub{Style.RESET_ALL}, {Fore.CYAN}requests{Style.RESET_ALL}. Needs PAT ({Fore.CYAN}GITHUB_TOKEN{Style.RESET_ALL} or prompt).
* {Style.BRIGHT}Upload{Style.RESET_ALL} ({Fore.GREEN}process{Style.RESET_ALL}/{Fore.MAGENTA}encrypt{Style.RESET_ALL}): Triggered by {Fore.YELLOW}--upload-to-github{Style.RESET_ALL}. Checks for existing release tag.
* {Style.BRIGHT}Download{Style.RESET_ALL} ({Fore.BLUE}decrypt{Style.RESET_ALL}): Optionally pulls layers if GitHub info present in DB.
* {Style.BRIGHT}Deletion{Style.RESET_ALL} ({Fore.RED}delete{Style.RESET_ALL}): {Fore.RED}Permanently deletes release assets{Style.RESET_ALL} corresponding to the DB entry ID.

{Style.BRIGHT}{Fore.MAGENTA}------------------ EXAMPLES -----------------{Style.RESET_ALL}
{Style.DIM}# Start DB server (separate terminal){Style.RESET_ALL}
$ {Fore.CYAN}python{Style.RESET_ALL} {Fore.GREEN}{DB_SERVER_SCRIPT}{Style.RESET_ALL}
{Style.DIM}# Process & upload{Style.RESET_ALL}
$ {Fore.CYAN}python{Style.RESET_ALL} {script_name} {Fore.GREEN}process{Style.RESET_ALL} {Fore.YELLOW}file.dat{Style.RESET_ALL} {Fore.CYAN}--upload-to-github{Style.RESET_ALL}
{Style.DIM}# Encrypt locally{Style.RESET_ALL}
$ {Fore.CYAN}python{Style.RESET_ALL} {script_name} {Fore.MAGENTA}encrypt{Style.RESET_ALL} {Fore.YELLOW}secret.zip{Style.RESET_ALL} {Fore.CYAN}-o{Style.RESET_ALL} {Fore.GREEN}./enc_out{Style.RESET_ALL}
{Style.DIM}# List active entries{Style.RESET_ALL}
$ {Fore.CYAN}python{Style.RESET_ALL} {script_name} {Fore.CYAN}list{Style.RESET_ALL}
{Style.DIM}# List trashed entries{Style.RESET_ALL}
$ {Fore.CYAN}python{Style.RESET_ALL} {script_name} {Fore.CYAN}list{Style.RESET_ALL} {Fore.YELLOW}--trash{Style.RESET_ALL}
{Style.DIM}# Decrypt/reconstruct (prompts for ID), pull from GH{Style.RESET_ALL}
$ {Fore.CYAN}python{Style.RESET_ALL} {script_name} {Fore.BLUE}decrypt{Style.RESET_ALL} {Fore.CYAN}-o{Style.RESET_ALL} {Fore.YELLOW}rebuilt.dat{Style.RESET_ALL}
{Style.DIM}# Delete entry (prompts ID, confirms GH asset deletion){Style.RESET_ALL}
$ {Fore.CYAN}python{Style.RESET_ALL} {script_name} {Fore.RED}delete{Style.RESET_ALL}

{Style.BRIGHT}{Fore.MAGENTA}---------------- DEPENDENCIES --------------{Style.RESET_ALL}
* {Fore.GREEN}Required{Style.RESET_ALL}: {Fore.CYAN}requests{Style.RESET_ALL}
* {Fore.GREEN}Required for {Fore.MAGENTA}encrypt{Style.RESET_ALL}/{Fore.BLUE}decrypt{Style.RESET_ALL} modes: {Fore.CYAN}cryptography{Style.RESET_ALL}
* {Fore.GREEN}Required for GitHub features{Style.RESET_ALL}: {Fore.CYAN}PyGithub{Style.RESET_ALL}
* {Fore.YELLOW}Optional{Style.RESET_ALL} (Recommended): {Fore.CYAN}colorama{Style.RESET_ALL}, {Fore.CYAN}tqdm{Style.RESET_ALL}, {Fore.CYAN}python-dotenv{Style.RESET_ALL}, {Fore.CYAN}pytz{Style.RESET_ALL}, {Fore.CYAN}rich{Style.RESET_ALL}
* DB Server ({Fore.GREEN}{DB_SERVER_SCRIPT}{Style.RESET_ALL}) Requires: {Fore.CYAN}Flask{Style.RESET_ALL}

{Style.BRIGHT}{Fore.MAGENTA}-------------------------------------------{Style.RESET_ALL}
"""
    )
    subparsers = parser.add_subparsers(dest="mode", required=True, title=f'{Style.BRIGHT}Available Modes{Style.RESET_ALL}',
        description=f'Select mode. Use {Fore.CYAN}<mode> --help{Style.RESET_ALL} for options.', help=f'{Fore.YELLOW}Operation mode{Style.RESET_ALL}')

    # --- Parent for process/encrypt ---
    parent_proc_enc = argparse.ArgumentParser(add_help=False)
    parent_proc_enc.add_argument("source_file", metavar="SOURCE_FILE", type=str, help=f"{Fore.YELLOW}Source file{Style.RESET_ALL} to process/encrypt.")
    parent_proc_enc.add_argument("-p", "--part-size", type=int, default=DEFAULT_PART_SIZE_MB, metavar="MB", help=f"Target part size ({Fore.CYAN}MB{Style.RESET_ALL}).")
    parent_proc_enc.add_argument("-w", "--workers", type=int, default=DEFAULT_WORKERS, metavar="N", help=f"Concurrent {Fore.CYAN}workers{Style.RESET_ALL} for processing.")
    parent_proc_enc.add_argument("-b", "--buffer-size", type=int, default=DEFAULT_HASH_BUFFER_MB, metavar="MB", help=f"I/O buffer ({Fore.CYAN}MB{Style.RESET_ALL}).")
    parent_proc_enc.add_argument("-o", "--output-dir", metavar="PATH", type=str, default=None, help=f"Output directory for parts/JSON. {Fore.YELLOW}(Default: auto-named dir in CWD){Style.RESET_ALL}")
    parent_proc_enc.add_argument("--upload-to-github", action="store_true", help=f"{Fore.YELLOW}Upload parts{Style.RESET_ALL} to GitHub Release.")

    # --- 'process' ---
    subparsers.add_parser("process", parents=[parent_proc_enc], help=f"Split file. Logs to DB. {Fore.YELLOW}Optionally uploads parts+JSON to GH.{Style.RESET_ALL}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=f"{Fore.GREEN}{Style.BRIGHT}Process Mode{Style.RESET_ALL}").set_defaults(func=process_main)

    # --- 'encrypt' ---
    subparsers.add_parser("encrypt", parents=[parent_proc_enc], help=f"{Fore.MAGENTA}Encrypt{Style.RESET_ALL} parts. Logs to DB. {Fore.YELLOW}Optionally uploads enc parts ONLY to GH.{Style.RESET_ALL}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=f"{Fore.MAGENTA}{Style.BRIGHT}Encrypt Mode{Style.RESET_ALL}").set_defaults(func=encrypt_main)

    # --- 'decrypt' ---
    parser_decrypt = subparsers.add_parser("decrypt", help=f"Reconstruct file from DB entry. {Fore.YELLOW}Optionally pulls from GH.{Style.RESET_ALL}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=f"{Fore.BLUE}{Style.BRIGHT}Decrypt/Reconstruct Mode{Style.RESET_ALL}")
    parser_decrypt.add_argument("-o", "--output-file", metavar="PATH", type=str, default=None, help=f"Path for final {Fore.YELLOW}reconstructed file{Style.RESET_ALL}. {Fore.YELLOW}(Default: original name in CWD){Style.RESET_ALL}")
    parser_decrypt.add_argument("-w", "--workers", type=int, default=DEFAULT_WORKERS, metavar="N", help=f"Concurrent {Fore.CYAN}workers{Style.RESET_ALL} for extraction/assembly.")
    parser_decrypt.add_argument("--download-workers", type=int, default=DEFAULT_DOWNLOAD_WORKERS, metavar="N", help=f"Concurrent {Fore.CYAN}workers{Style.RESET_ALL} for GitHub pull (max: {MAX_DOWNLOAD_WORKERS}).")
    parser_decrypt.add_argument("-b", "--buffer-size", type=int, default=DEFAULT_HASH_BUFFER_MB, metavar="MB", help=f"Verification hash buffer ({Fore.CYAN}MB{Style.RESET_ALL}).")
    parser_decrypt.set_defaults(func=decrypt_main)

    # --- ADDED 'list' Subparser ---
    parser_list = subparsers.add_parser("list", help=f"List entries from DB ({Fore.YELLOW}--trash{Style.RESET_ALL} for deleted).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=f"{Fore.CYAN}{Style.BRIGHT}List Mode{Style.RESET_ALL}")
    parser_list.add_argument("--trash", action="store_true", help=f"Show {Fore.YELLOW}trashed{Style.RESET_ALL} entries instead of active ones.")
    parser_list.set_defaults(func=list_main)

    # --- ADDED 'delete' Subparser ---
    parser_delete = subparsers.add_parser("delete", help=f"{Fore.RED}Delete{Style.RESET_ALL} entry (moves to trash, {Fore.RED}DEL GH assets{Style.RESET_ALL}).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=f"{Fore.RED}{Style.BRIGHT}Delete Mode{Style.RESET_ALL}")
    # No specific arguments needed for delete currently, just uses interactive prompts
    parser_delete.set_defaults(func=delete_main)
    parser_get_dl = subparsers.add_parser("get-download",
        help=f"List active entries and print GitHub asset {Fore.YELLOW}download URLs{Style.RESET_ALL} for a selected ID.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=f"{Fore.GREEN}{Style.BRIGHT}Get Download URLs Mode{Style.RESET_ALL}")
    # No specific arguments needed for get-download initially, uses interactive prompts
    parser_get_dl.set_defaults(func=get_download_main) # Assign the new handler function

    return parser

    return parser
 
# --- DB Server Management Functions ---

def is_db_server_running(host: str, port: int) -> bool:
    """Checks if the DB server is accepting connections and responds to /status."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1) # Don't wait too long for initial connection
    try:
        if sock.connect_ex((host, port)) == 0:
            # Port is open, now check if it's our server via the /status endpoint
            try:
                response = requests.get(f"http://{host}:{port}/status", timeout=2)
                # Check for 200 OK and specific content
                return response.status_code == 200 and response.json().get("status") == "ok"
            except (requests.exceptions.RequestException, json.JSONDecodeError):
                # Port open, but couldn't connect via HTTP or get expected JSON
                print_log("DEBUG", f"Port {port} open but /status check failed.")
                return False
        else:
            # Connection refused
            return False
    except socket.error as e:
        print_log("DEBUG", f"Socket error checking server status: {e}")
        return False
    finally:
        sock.close()

def start_db_server(script_path: str, python_exe: str = sys.executable) -> Optional[subprocess.Popen]:
    """Starts the DB server script as a background process."""
    if not os.path.isfile(script_path):
        print_log("ERROR", f"DB Server script not found: {script_path}")
        return None

    print_log("INFO", f"Attempting to start DB Server: {python_exe} {script_path}")
    try:
        kwargs = {}
        if platform.system() == "Windows":
            # Prevent console window from popping up on Windows
            kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
        else:
            # Detach on Unix-like systems
            kwargs['start_new_session'] = True

        # Make sure environment variables (like DB_SERVER_FILE if set) are passed
        env = os.environ.copy()

        process = subprocess.Popen(
            [python_exe, script_path],
            env=env,
            stdout=subprocess.PIPE, # Capture stdout for potential debugging
            stderr=subprocess.PIPE, # Capture stderr
            **kwargs
        )
        print_log("INFO", f"DB Server process started (PID: {process.pid}). Waiting briefly...")
        return process # Return the Popen object
    except OSError as e:
        print_log("ERROR", f"Failed to start DB server process: {e}")
    except Exception as e:
        print_log("ERROR", f"Unexpected error starting DB server: {e}")
    return None


# --- Script Execution Entry Point ---
if __name__ == "__main__":
    # Print dependency warnings
    if not CRYPTOGRAPHY_AVAILABLE: print(f"{Fore.YELLOW}Warning: 'cryptography' not found. 'encrypt'/'decrypt' modes may fail.{Style.RESET_ALL}")
    if not PYGITHUB_AVAILABLE: print(f"{Fore.YELLOW}Warning: 'PyGithub' not found. GitHub features disabled.{Style.RESET_ALL}")
    if not REQUESTS_AVAILABLE: print(f"{Fore.RED}ERROR: 'requests' library not found. This script requires it for DB server communication.{Style.RESET_ALL}"); sys.exit(1)

    # --- Check and Start DB Server ---
    print_log("INFO", f"Checking if DB server is running at {DB_SERVER_URL}...")
    server_process_obj: Optional[subprocess.Popen] = None # To hold the Popen object if we start it
    if not is_db_server_running(DB_SERVER_HOST, DB_SERVER_PORT):
        print_log("WARN", "DB Server not detected. Attempting to start...")
        server_process_obj = start_db_server(DB_SERVER_SCRIPT)
        if server_process_obj:
            time.sleep(3) # Give server time to initialize
            if not is_db_server_running(DB_SERVER_HOST, DB_SERVER_PORT):
                print_log("ERROR", "DB Server failed to start or respond correctly after launch attempt.")
                # Try to read output from the failed process
                try:
                    stdout, stderr = server_process_obj.communicate(timeout=1)
                    if stdout: print_log("DEBUG", f"Server stdout:\n{stdout.decode(errors='ignore')}")
                    if stderr: print_log("ERROR", f"Server stderr:\n{stderr.decode(errors='ignore')}")
                except subprocess.TimeoutExpired:
                    print_log("WARN", "Reading server output timed out.")
                    server_process_obj.kill() # Terminate if stuck
                except Exception as comm_err:
                     print_log("WARN", f"Error getting server process output: {comm_err}")
                sys.exit(1)
            else:
                print_log("SUCCESS", "DB Server started and responded successfully.")
                # Note: We don't automatically stop the server we started. It runs in the background.
        else:
            print_log("ERROR", "Failed to initiate DB server startup command.")
            sys.exit(1)
    else:
        print_log("INFO", "DB Server is running.")
    # --- DB Server Check Complete ---


    parser = setup_arg_parser()
    if len(sys.argv) == 1: parser.print_help(sys.stderr); sys.exit(1)

    try:
        args = parser.parse_args() # Full parse

        # Basic Input Validation
        if hasattr(args, 'workers') and args.workers <= 0: raise ValueError("Workers must be > 0.")
        if hasattr(args, 'download_workers'):
            if args.download_workers <= 0: raise ValueError("Download workers must be > 0.")
            if args.download_workers > MAX_DOWNLOAD_WORKERS:
                print_log("WARN", f"Requested download workers ({args.download_workers}) exceeds max ({MAX_DOWNLOAD_WORKERS}). Using {MAX_DOWNLOAD_WORKERS}.")
                args.download_workers = MAX_DOWNLOAD_WORKERS
            if args.download_workers < 1: # Ensure at least 1 worker
                args.download_workers = 1

        if hasattr(args, 'buffer_size') and args.buffer_size <= 0: raise ValueError("Buffer size must be > 0.")

        # Mode-Specific Pre-flight Checks
        if args.mode in ["process", "encrypt"]:
            if not os.path.isfile(args.source_file): raise FileNotFoundError(f"Source file not found: {args.source_file}")
            if args.part_size <= 0: raise ValueError("Part size must be > 0.")
            if args.upload_to_github and not PYGITHUB_AVAILABLE: raise ImportError("PyGithub required for --upload-to-github.")
            if args.mode == 'encrypt' and not CRYPTOGRAPHY_AVAILABLE: raise ImportError("'encrypt' mode requires 'cryptography'.")
        elif args.mode == "decrypt":
             # Checks done within decrypt_main, especially crypto check after selection
             pass
        elif args.mode == "delete" and not PYGITHUB_AVAILABLE:
            print_log("WARN", "PyGithub not found. Cannot delete GitHub assets if needed for the selected entry.")
        elif args.mode in ["decrypt", "list"]: pass # Checks done within functions

        # Execute Main Function
        args.func(args)

    except (ValueError, FileNotFoundError, ImportError) as e: print_log("ERROR", f"Config/Input Error: {e}"); sys.exit(1)
    except requests.exceptions.ConnectionError as conn_e:
        print_log("ERROR", f"DB Client Error: Could not connect to the database server at {DB_SERVER_URL}. Is it running?")
        print_log("DEBUG", f"Connection Error Details: {conn_e}")
        sys.exit(1)
    except requests.exceptions.RequestException as req_e:
        print_log("ERROR", f"DB Client Error: Request to DB server failed: {req_e}")
        if req_e.response is not None:
             print_log("DEBUG", f"Server Response: {req_e.response.status_code} - {req_e.response.text}")
        sys.exit(1)
    except SystemExit as e: sys.exit(e.code) # Allow clean exits
    except KeyboardInterrupt: print_log("WARN", "\nOperation cancelled by user (Ctrl+C)."); sys.exit(130)
    except (RateLimitExceededException, ConnectionAbortedError) as gh_rl_e: print_log("ERROR", f"GitHub API Error: {gh_rl_e}"); sys.exit(1)
    except GithubException as gh_e: print_log("ERROR", f"GitHub API error (Status: {gh_e.status}): {gh_e.data.get('message', 'Unknown')}"); print_log("DEBUG", f"GH Error Data: {gh_e.data}"); sys.exit(1)
    except ConnectionError as conn_e: print_log("ERROR", f"Network connection error (non-DB): {conn_e}"); sys.exit(1)
    except Exception as main_exception:
        print_log("ERROR", f"Critical unexpected error: {main_exception}")
        print(f"{Fore.RED}{Style.BRIGHT}--- TRACEBACK ---{Style.RESET_ALL}")
        print(f"{Fore.RED}{traceback.format_exc()}{Style.RESET_ALL}")
        print(f"{Fore.RED}{Style.BRIGHT}-----------------{Style.RESET_ALL}")
        sys.exit(1)

