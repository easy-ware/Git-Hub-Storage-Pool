# Git-Hub-Storage-Pool (easy-ware)

**Version:** 1.0.0

Utilize GitHub Releases as a storage backend for large files by splitting them into manageable parts, optionally encrypting them, and logging metadata locally.

## Overview

This project provides a command-line tool (`storage_pool.py`) to interact with a local database server (`db_server.py`) for managing large files. It allows you to:

1.  **Split:** Break down large files into smaller parts.
2.  **Encrypt:** Optionally encrypt these parts using AES-GCM encryption.
3.  **Upload:** Store these parts as assets on a private GitHub Release.
4.  **Log:** Record metadata (including filenames, SHAs, part info, GitHub location, and decryption keys if encrypted) in a local SQLite database managed by a Flask server.
5.  **List:** View active or trashed file entries stored in the database.
6.  **Download & Reconstruct:** Retrieve parts (optionally pulling from GitHub using `aria2c`) and reconstruct the original file, decrypting if necessary.
7.  **Delete:** Move database entries to a 'trash' table and permanently delete associated assets and the release from GitHub.
8.  **Get URLs:** Retrieve direct download URLs for assets stored on GitHub releases.

## Architecture

* **Client:** `storage_pool.py` - The main command-line interface users interact with. It handles file processing, encryption, GitHub API calls, and communication with the DB server.
* **Server:** `db_server.py` - A simple Flask application that manages the SQLite database (`file_processor.db` by default). It provides API endpoints for the client to add, query, and delete log entries.
* **Database Logic:** `database/database.py` - Contains functions for initializing the SQLite database schema and performing CRUD operations.
* **GitHub:** Used as the storage backend. File parts are uploaded as assets to specific, auto-generated releases within a designated repository.
* **aria2c:** An external download utility used for efficient parallel downloading of file parts from GitHub during the `decrypt` process.

## Features

* **Multiple Modes:** `process` (split), `encrypt` (split & encrypt), `decrypt` (reconstruct/decrypt), `list` (view DB entries), `delete` (remove entry & GH assets), `get-download` (get asset URLs).
* **Client-Server Database:** Centralized logging via a local Flask server.
* **GitHub Integration:** Leverages GitHub Releases for storing file parts. Handles repository creation (if needed), release creation, and asset upload/deletion. Uses sanitized tags to comply with GitHub rules.
* **Encryption:** Secure AES-GCM 256-bit encryption for parts in `encrypt` mode. Keys are stored (base64 encoded) in the local database ONLY.
* **Parallel Processing:** Uses `concurrent.futures` for parallel splitting/encryption/decryption.
* **Parallel Downloads:** Uses `aria2c` for accelerated downloads from GitHub (if installed).
* **Integrity Checks:** Uses SHA-256 checksums to verify original files, encrypted parts (before decryption), and final reconstructed files.
* **Configuration:** Uses `.env` file (`config/system_config.conf`) for settings like GitHub token and DB server parameters.
* **Interactive Prompts:** User-friendly prompts for confirmation and input (requires `rich`).
* **Detailed Logging:** Colored console output for progress and status updates (requires `colorama`, `tqdm`).

## Installation

1.  **Prerequisites:**
    * Python 3.7+
    * `pip` (Python package installer)
    * Git

2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/easy-ware/Git-Hub-Storage-Pool.git
    cd Git-Hub-Storage-Pool
    ```

3.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements/req.txt
    ```
    This installs necessary libraries like `requests`, `PyGithub`, `cryptography`, `Flask`, `rich`, `colorama`, `tqdm`, `python-dotenv`, `pytz`.

4.  **Install `aria2c` (Required for GitHub Downloads in `decrypt` mode):**
    * **Linux (Debian/Ubuntu):**
        ```bash
        sudo apt update && sudo apt install aria2c
        ```
        *(The script might attempt automatic installation via `apt` if `aria2c` is not found on Linux, prompting for sudo password).*
    * **macOS (using Homebrew):**
        ```bash
        brew install aria2
        ```
    * **Windows:**
        * Download the latest `aria2c.exe` from the [aria2 Releases page](https://github.com/aria2/aria2/releases).
        * Create a folder named `drivers` inside the `Git-Hub-Storage-Pool` directory.
        * Place the downloaded `aria2c.exe` file inside the `drivers` folder. The script expects it at `./drivers/aria2c.exe`.

5.  **Configuration:**
    * Create a configuration file: `config/system_config.conf` (you might need to create the `config` directory first).
    * Add the following environment variables to the file:

        ```dotenv
        # config/system_config.conf

        # Required for GitHub Upload/Download/Delete operations
        # Generate a Personal Access Token (PAT) from GitHub with 'repo' scope.
        # If creating repositories automatically, ensure appropriate permissions.
        GITHUB_TOKEN=YOUR_GITHUB_PERSONAL_ACCESS_TOKEN

        # Optional: Database Server Configuration (Defaults shown)
        # DB_SERVER_HOST=127.0.0.1
        # DB_SERVER_PORT=8034
        # DB_SERVER_FILE=file_processor.db # Filename within ./database_files/

        DRIVER_NAME = "aria2c.exe" # Give the correct and complete name with extension of downloaded driver
        ```
    * **Important:** Replace `YOUR_GITHUB_PERSONAL_ACCESS_TOKEN` with your actual GitHub PAT. Keep this file secure and consider adding it to your `.gitignore` if you fork the repository.

## Usage

**1. Start the Database Server:**

* Open a separate terminal in the project directory.
* Run the server script:
    ```bash
    python database/db_server.py
    ```
* Keep this server running in the background while using the main `storage_pool.py` script. The main script will attempt to automatically start the server if it's not detected, but running it manually is recommended.

**2. Run the Main Script (`storage_pool.py`):**

* Use another terminal in the project directory.
* The script uses different modes specified as the first argument.

    ```bash
    python storage_pool.py <mode> [options...]
    ```

**Modes:**

* **`process` (Split File):**
    ```bash
    python storage_pool.py process <source_file> [options]
    ```
    * Splits `<source_file>` into parts.
    * Logs metadata and part SHAs to the database.
    * `--part-size MB`: Set part size (default: 1800 MB).
    * `--workers N`: Number of parallel workers (default: 2).
    * `--output-dir PATH`: Specify output directory (default: `./<filename>_split_process`).
    * `--upload-to-github`: Upload parts **and** the metadata JSON file to a GitHub release. Prompts for `owner/repo` and uses `GITHUB_TOKEN`.

* **`encrypt` (Split & Encrypt File):**
    ```bash
    python storage_pool.py encrypt <source_file> [options]
    ```
    * Splits `<source_file>` and encrypts each part.
    * Logs metadata, encrypted part SHAs, and **decryption keys** to the database.
    * Options similar to `process`.
    * `--upload-to-github`: Uploads **only the encrypted parts** (NOT the JSON with keys) to a GitHub release.

* **`list` (List Entries):**
    ```bash
    python storage_pool.py list [options]
    ```
    * Displays entries from the database.
    * `--trash`: Show entries in the 'trash' table instead of active ones.

* **`decrypt` (Reconstruct / Decrypt File):**
    ```bash
    python storage_pool.py decrypt [options]
    ```
    * Lists active entries from the database and prompts for an ID to reconstruct.
    * If the entry has GitHub info, prompts to pull parts from GitHub (requires `aria2c`). Otherwise, uses the local `output_directory` stored in the DB.
    * Reconstructs the original file, decrypting parts if the mode was `encrypt`.
    * Verifies the final file's SHA-256 against the original.
    * `--output-file PATH`: Specify the path for the reconstructed file (default: original filename in CWD).
    * `--workers N`: Workers for decryption/assembly (default: 2).
    * `--download-workers N`: Workers for parallel GitHub downloads via `aria2c` (default: 2, max: 3).

* **`delete` (Delete Entry):**
    ```bash
    python storage_pool.py delete
    ```
    * Lists active entries and prompts for an ID to delete.
    * **Moves the database entry to the 'trash' table.**
    * If the entry has GitHub info:
        * **PERMANENTLY DELETES** all associated assets from the GitHub release.
        * **PERMANENTLY DELETES** the GitHub release itself.
    * Requires multiple confirmations, especially for GitHub deletion. **This action is irreversible on GitHub.**

* **`get-download` (Get GitHub Download URLs):**
    ```bash
    python storage_pool.py get-download
    ```
    * Lists active entries that have associated GitHub information.
    * Prompts for an ID.
    * Uses the GitHub API to resolve and print the direct download URLs for each asset (part) in the selected release.

**General Options:**

* `-h`, `--help`: Show help message. Use `python storage_pool.py <mode> --help` for mode-specific help.
* `-b MB`, `--buffer-size MB`: I/O buffer size for hashing/verification (default: 4 MB).

## How It Works

1.  **Splitting/Encryption:** The source file is read in chunks (`--part-size`). For `process`, chunks are saved directly. For `encrypt`, each chunk is encrypted using AES-GCM with a unique, randomly generated key and nonce. The nonce is prepended to the ciphertext.
2.  **Metadata:** Information about the original file (name, size, SHA256), processing settings, number of parts, and part-specific details (SHA, filename, key for `encrypt` mode) is compiled into a JSON structure.
3.  **Database Logging:** The client sends the metadata JSON (and GitHub info if applicable) to the `db_server.py` via HTTP POST request. The server stores this information in the `processed_files` table in the SQLite database. Decryption keys (if any) are stored here.
4.  **GitHub Upload (Optional):**
    * A GitHub repository and Personal Access Token (`GITHUB_TOKEN`) are required.
    * A unique release tag is generated based on the mode, filename, and SHA (e.g., `file-encrypt-myfile.zip-a1b2c3d4`). This tag is sanitized to be GitHub-compatible.
    * The script checks if a release with the sanitized tag already exists. If not, it creates one.
    * Parts are uploaded as assets to this release. For `process` mode, the metadata JSON is also uploaded. For `encrypt` mode, the JSON (containing keys) is **NOT** uploaded.
    * The GitHub repository name and the *sanitized* release tag used are stored in the database along with the original proposed tag.
5.  **Reconstruction (`decrypt`):**
    * The user selects an entry ID from the database list.
    * The script retrieves the full metadata from the DB server.
    * It determines the source of the parts: either pulls from the linked GitHub release (using `aria2c` if available and confirmed) or uses the local directory path stored in the DB.
    * Parts are read sequentially. If encrypted, they are decrypted using the key retrieved from the database metadata.
    * Decrypted/plaintext data is written to the output file.
    * The final reconstructed file's SHA-256 is verified against the original SHA stored in the metadata.
6.  **Deletion (`delete`):**
    * The user selects an entry ID.
    * If GitHub info exists, the script uses the GitHub API to delete all assets from the corresponding release and then deletes the release itself. **This is permanent.**
    * If GitHub deletion is successful (or not applicable), the script requests the DB server to move the entry from `processed_files` to the `trash` table via HTTP DELETE request.

## Error Handling & Troubleshooting

* **DB Server Not Running:** Ensure `python database/db_server.py` is running in a separate terminal. The client script will attempt to start it but might fail depending on permissions or environment.
* **Connection Refused (DB Server):** Check if the `DB_SERVER_HOST` and `DB_SERVER_PORT` in `config/system_config.conf` (or defaults) match where the server is running. Check firewalls.
* **Missing Dependencies:** Run `pip install -r req.txt`. Ensure `cryptography` is installed for `encrypt`/`decrypt`. Ensure `PyGithub` is installed for GitHub features.
* **`aria2c` Not Found:** Make sure `aria2c` is installed and accessible in your system's PATH (Linux/macOS) or located in the `./drivers/` directory (Windows). See Installation section.
* **GitHub Errors:**
    * **401 Bad Credentials:** Invalid `GITHUB_TOKEN`. Generate a new PAT with `repo` scope.
    * **404 Not Found:** Repository name incorrect, or token lacks permission for the private repo. Release tag not found during download/delete.
    * **403 Forbidden/Rate Limit:** GitHub API rate limit exceeded. Wait for the reset period (usually an hour). Check token permissions.
    * **422 Unprocessable Entity:** Often occurs if a release tag/asset already exists when trying to create it (the script handles this by skipping or warning). Can also indicate invalid characters in names/tags (though sanitization aims to prevent this).
* **File Not Found:** Check source file paths and output directory paths. Ensure the local directory referenced in the DB for `decrypt` mode still exists if not pulling from GitHub.
* **SHA Mismatch:** Indicates corruption during download, storage, or reconstruction. Check the source parts.
* **Decryption Failed (InvalidTag):** Wrong key used, or the encrypted part is corrupted. Ensure the correct database entry was selected.

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.

## License

MIT License




# Team Easy-ware