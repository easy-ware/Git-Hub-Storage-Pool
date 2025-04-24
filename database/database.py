# -*- coding: utf-8 -*-
import sqlite3
import json
import os
import datetime
import threading
from typing import Optional, Dict, Any, List

# Use a lock for thread-safe database access if the server uses threads
db_lock = threading.Lock()

# --- Logging Placeholder ---
def print_db_log(level, message):
    """Basic logging for the database module."""
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + "Z"
    print(f"[{timestamp}][DB_MODULE][{level}] {message}") # Changed prefix for clarity

# --- Database Functions ---

def _add_column_if_not_exists(cursor: sqlite3.Cursor, table_name: str, column_name: str, column_type: str):
    """Helper to add a column only if it doesn't exist."""
    try:
        cursor.execute(f"SELECT {column_name} FROM {table_name} LIMIT 1;")
        print_db_log("DEBUG", f"Column '{column_name}' already exists in table '{table_name}'.")
    except sqlite3.OperationalError as e:
        if f"no such column: {column_name}" in str(e):
            print_db_log("INFO", f"Column '{column_name}' not found in '{table_name}'. Adding column...")
            try:
                cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type};")
                print_db_log("INFO", f"Successfully added column '{column_name}' to table '{table_name}'.")
            except sqlite3.Error as alter_e:
                 print_db_log("ERROR", f"Failed to add column '{column_name}' to '{table_name}': {alter_e}")
                 raise # Re-raise the error to stop initialization if alter fails
        else:
             raise e # Re-raise other operational errors


def init_db(db_path: str):
    """Initializes the SQLite database and tables (processed_files, trash), adding new tag columns if needed."""
    print_db_log("INFO", f"Initializing database connection: {db_path}")
    conn = None
    try:
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            print_db_log("INFO", f"Created directory for database: {db_dir}")

        # Use longer timeout for potential schema changes
        conn = sqlite3.connect(db_path, timeout=20, check_same_thread=False)
        cursor = conn.cursor()

        # --- Main Table ---
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS processed_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_filename TEXT NOT NULL,
            original_sha256 TEXT NOT NULL,
            file_size_bytes INTEGER NOT NULL,
            mode TEXT NOT NULL CHECK(mode IN ('process', 'encrypt')),
            processing_datetime_utc TEXT NOT NULL,
            number_of_parts INTEGER NOT NULL,
            output_directory TEXT NOT NULL,
            github_repo TEXT,
            github_release_tag TEXT, -- Kept for backward compatibility/reference
            metadata_json TEXT NOT NULL
            -- New columns will be added below if they don't exist
        );
        """)
        print_db_log("INFO", "Ensured base table 'processed_files' exists.")
        
        # --- Add new columns to processed_files if they don't exist ---
        _add_column_if_not_exists(cursor, "processed_files", "original_release_tag", "TEXT")
        _add_column_if_not_exists(cursor, "processed_files", "sanitized_release_tag", "TEXT")
        # --- End column adds for processed_files ---

        # --- Trash Table ---
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS trash (
            id INTEGER PRIMARY KEY, -- Use the original ID from processed_files
            original_filename TEXT NOT NULL,
            original_sha256 TEXT NOT NULL,
            file_size_bytes INTEGER NOT NULL,
            mode TEXT NOT NULL CHECK(mode IN ('process', 'encrypt')),
            processing_datetime_utc TEXT NOT NULL, -- When originally processed
            deleted_datetime_utc TEXT NOT NULL,   -- When moved to trash
            number_of_parts INTEGER NOT NULL,
            output_directory TEXT NOT NULL,
            github_repo TEXT,
            github_release_tag TEXT, -- Kept for backward compatibility/reference
            metadata_json TEXT NOT NULL
            -- New columns will be added below if they don't exist
        );
        """)
        print_db_log("INFO", "Ensured base table 'trash' exists.")

        # --- Add new columns to trash if they don't exist ---
        _add_column_if_not_exists(cursor, "trash", "original_release_tag", "TEXT")
        _add_column_if_not_exists(cursor, "trash", "sanitized_release_tag", "TEXT")
        # --- End column adds for trash ---


        conn.commit()
        return True
    except sqlite3.Error as e:
        print_db_log("ERROR", f"Database error during initialization: {e}")
        if conn: conn.rollback()
        return False
    except OSError as e:
        print_db_log("ERROR", f"OS error creating directory for DB: {e}")
        return False
    except Exception as e:
        print_db_log("ERROR", f"Unexpected error during DB init: {e}")
        if conn: conn.rollback()
        return False
    finally:
        if conn:
            conn.close()
            print_db_log("DEBUG", "Database connection closed after init.")


def add_log_entry(db_path: str, metadata: Dict[str, Any], github_info: Optional[Dict[str, str]] = None) -> Optional[int]:
    """
    Adds a record to 'processed_files', including original and sanitized tags. 
    Returns new row ID or None.
    """
    print_db_log("INFO", f"Attempting to add log entry for SHA: {metadata.get('original_sha256', 'N/A')[:8]}...")

    required_keys = ["original_filename", "original_sha256", "file_size_bytes", "mode",
                     "creation_datetime_utc", "number_of_parts", "settings"]
    if not all(key in metadata for key in required_keys) or not isinstance(metadata.get("settings"), dict):
        print_db_log("ERROR", "Metadata missing required keys for DB logging.")
        return None

    output_dir = metadata["settings"].get("output_directory")
    if not output_dir:
        print_db_log("ERROR", "Output directory missing in metadata for DB logging.")
        return None

    # --- MODIFICATION: Extract new tag info ---
    repo_name = github_info.get("repo_full_name") if github_info else None
    # Keep old tag for potential backward compatibility or direct reference if needed
    legacy_release_tag = github_info.get("release_tag") if github_info else None 
    original_tag = github_info.get("original_release_tag") if github_info else None
    sanitized_tag = github_info.get("sanitized_release_tag") if github_info else None
    
    # Prefer the new sanitized tag for the main reference field if available
    final_tag_to_store = sanitized_tag if sanitized_tag else legacy_release_tag
    # --- END MODIFICATION ---

    try:
        metadata_json_str = json.dumps(metadata)
    except TypeError as e:
        print_db_log("ERROR", f"Failed to serialize metadata to JSON for DB: {e}")
        return None

    conn = None
    last_id = None
    try:
        with db_lock:
            conn = sqlite3.connect(db_path, timeout=10, check_same_thread=False)
            cursor = conn.cursor()
            
            # --- MODIFICATION: Include new columns in INSERT ---
            cursor.execute("""
            INSERT INTO processed_files (
                original_filename, original_sha256, file_size_bytes, mode,
                processing_datetime_utc, number_of_parts, output_directory,
                github_repo, github_release_tag, metadata_json,
                original_release_tag, sanitized_release_tag 
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata["original_filename"], metadata["original_sha256"],
                metadata["file_size_bytes"], metadata["mode"],
                metadata["creation_datetime_utc"], metadata["number_of_parts"],
                output_dir, 
                repo_name, 
                final_tag_to_store, # Store the sanitized (or legacy) tag here
                metadata_json_str,
                original_tag,      # Store the original proposed tag
                sanitized_tag      # Store the final sanitized tag used by GH
            ))
            # --- END MODIFICATION ---
            
            conn.commit()
            last_id = cursor.lastrowid
            if last_id:
                print_db_log("SUCCESS", f"Successfully added log entry ID {last_id} to 'processed_files' for '{metadata['original_filename']}'.")
            else:
                print_db_log("WARN", f"INSERT completed but no last row ID obtained for '{metadata['original_filename']}'.")

    except sqlite3.Error as e:
        print_db_log("ERROR", f"Database error adding log entry: {e}")
        last_id = None
    except Exception as e:
        print_db_log("ERROR", f"Unexpected error adding DB log entry: {e}")
        last_id = None
    finally:
        if conn: conn.close()

    return last_id


def get_all_entries(db_path: str) -> List[Dict[str, Any]]:
    """Queries 'processed_files' for active entries, including new tag fields."""
    print_db_log("INFO", f"Querying database for active entries: {db_path}")
    conn = None
    entries = []
    try:
        with db_lock:
            conn = sqlite3.connect(db_path, timeout=10, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            # --- MODIFICATION: Select new columns ---
            cursor.execute("""
            SELECT id, original_filename, original_sha256, file_size_bytes, mode,
                   processing_datetime_utc, github_repo, 
                   github_release_tag, original_release_tag, sanitized_release_tag, -- Added tags
                   number_of_parts, output_directory
            FROM processed_files ORDER BY processing_datetime_utc ASC;
            """)
            # --- END MODIFICATION ---
            entries = [dict(row) for row in cursor.fetchall()]
            print_db_log("INFO", f"Found {len(entries)} active entries.")
    except sqlite3.Error as e:
        print_db_log("ERROR", f"Database error querying active entries: {e}")
    except Exception as e:
        print_db_log("ERROR", f"Unexpected error querying active entries: {e}")
    finally:
        if conn: conn.close()
    return entries


def get_trash_entries(db_path: str) -> List[Dict[str, Any]]:
    """Queries 'trash' for deleted entries, including new tag fields."""
    print_db_log("INFO", f"Querying database for trash entries: {db_path}")
    conn = None
    entries = []
    try:
        with db_lock:
            conn = sqlite3.connect(db_path, timeout=10, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            # --- MODIFICATION: Select new columns ---
            cursor.execute("""
            SELECT id, original_filename, original_sha256, file_size_bytes, mode,
                   processing_datetime_utc, deleted_datetime_utc, github_repo, 
                   github_release_tag, original_release_tag, sanitized_release_tag, -- Added tags
                   number_of_parts, output_directory
            FROM trash ORDER BY deleted_datetime_utc DESC;
            """)
            # --- END MODIFICATION ---
            entries = [dict(row) for row in cursor.fetchall()]
            print_db_log("INFO", f"Found {len(entries)} trash entries.")
    except sqlite3.Error as e:
        print_db_log("ERROR", f"Database error querying trash entries: {e}")
    except Exception as e:
        print_db_log("ERROR", f"Unexpected error querying trash entries: {e}")
    finally:
        if conn: conn.close()
    return entries


def get_entry_details(db_path: str, entry_id: int, from_trash: bool = False) -> Optional[Dict[str, Any]]:
    """Fetches full details for an entry, including new tag fields and parsed metadata."""
    table = "trash" if from_trash else "processed_files"
    print_db_log("INFO", f"Fetching details for entry ID: {entry_id} from table '{table}'")
    conn = None
    details = None
    try:
        with db_lock:
            conn = sqlite3.connect(db_path, timeout=10, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # --- MODIFICATION: Select new columns ---
            columns = ("id, original_filename, original_sha256, file_size_bytes, mode, "
                       "processing_datetime_utc, number_of_parts, output_directory, "
                       "github_repo, github_release_tag, original_release_tag, "
                       "sanitized_release_tag, metadata_json")
            if from_trash:
                columns += ", deleted_datetime_utc"
            # --- END MODIFICATION ---

            query = f"SELECT {columns} FROM {table} WHERE id = ?;"
            cursor.execute(query, (entry_id,))
            row = cursor.fetchone()

            if row:
                details = dict(row)
                try:
                    details['metadata'] = json.loads(details['metadata_json'])
                    # del details['metadata_json'] # Keep raw json for move_to_trash
                    print_db_log("DEBUG", f"Successfully fetched and parsed metadata JSON for ID {entry_id} from '{table}'.")
                except json.JSONDecodeError as json_e:
                    print_db_log("ERROR", f"Failed to parse metadata JSON from DB for ID {entry_id} in '{table}': {json_e}")
                    details = None 
                except Exception as e:
                    print_db_log("ERROR", f"Unexpected error parsing metadata JSON for ID {entry_id} in '{table}': {e}")
                    details = None 
            else:
                print_db_log("WARN", f"No database entry found with ID: {entry_id} in table '{table}'")
    except sqlite3.Error as e:
        print_db_log("ERROR", f"Database error fetching details for ID {entry_id} from '{table}': {e}")
    except Exception as e:
        print_db_log("ERROR", f"Unexpected error fetching DB details for ID {entry_id} from '{table}': {e}")
    finally:
        if conn: conn.close()
    return details


def move_to_trash(db_path: str, entry_id: int) -> bool:
    """Moves an entry from 'processed_files' to 'trash', including new tag fields."""
    print_db_log("INFO", f"Attempting to move entry ID {entry_id} to trash...")

    conn = None
    try:
        with db_lock:
            conn = sqlite3.connect(db_path, timeout=10, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # 1. Fetch the entry from processed_files, including new fields
            # --- MODIFICATION: Select new columns ---
            cursor.execute("""
            SELECT id, original_filename, original_sha256, file_size_bytes, mode,
                   processing_datetime_utc, number_of_parts, output_directory,
                   github_repo, github_release_tag, metadata_json,
                   original_release_tag, sanitized_release_tag 
            FROM processed_files WHERE id = ?;
            """, (entry_id,))
            # --- END MODIFICATION ---
            row_to_move = cursor.fetchone()

            if not row_to_move:
                print_db_log("WARN", f"Entry ID {entry_id} not found in 'processed_files'. Cannot move to trash.")
                return False

            # Convert row to dictionary
            entry_data = dict(row_to_move)
            deleted_time_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

            # 2. Insert into trash table, including new fields
            # --- MODIFICATION: Insert new columns ---
            cursor.execute("""
            INSERT INTO trash (
                id, original_filename, original_sha256, file_size_bytes, mode,
                processing_datetime_utc, deleted_datetime_utc, number_of_parts,
                output_directory, github_repo, github_release_tag, metadata_json,
                original_release_tag, sanitized_release_tag
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry_data['id'], entry_data['original_filename'], entry_data['original_sha256'],
                entry_data['file_size_bytes'], entry_data['mode'], entry_data['processing_datetime_utc'],
                deleted_time_utc, # The new deleted timestamp
                entry_data['number_of_parts'], entry_data['output_directory'],
                entry_data['github_repo'], 
                entry_data['github_release_tag'], # Legacy tag
                entry_data['metadata_json'], # Store the raw JSON string
                entry_data.get('original_release_tag'), # New original tag (use .get for safety)
                entry_data.get('sanitized_release_tag') # New sanitized tag (use .get for safety)
            ))
            # --- END MODIFICATION ---

            # 3. Delete from processed_files
            cursor.execute("DELETE FROM processed_files WHERE id = ?", (entry_id,))

            # 4. Commit Transaction
            conn.commit()
            print_db_log("SUCCESS", f"Entry ID {entry_id} successfully moved to 'trash' table.")
            return True

    except sqlite3.IntegrityError as e:
         print_db_log("ERROR", f"Database Integrity error moving ID {entry_id} to trash (maybe already in trash?): {e}")
         return False # Likely already exists in trash if PK constraint fails
    except sqlite3.Error as e:
        print_db_log("ERROR", f"Database error moving entry ID {entry_id} to trash: {e}")
        if conn: conn.rollback() # Rollback on error
        return False
    except Exception as e:
        print_db_log("ERROR", f"Unexpected error moving entry ID {entry_id} to trash: {e}")
        if conn: conn.rollback()
        return False
    finally:
        if conn: conn.close()