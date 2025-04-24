# -*- coding: utf-8 -*-
import os
import sys
from flask import Flask, request, jsonify
import database.database as database  # Import the database logic
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.getcwd() , "config" , "system_config.conf"))

# --- Configuration (remains the same) ---
DEFAULT_DB_FILENAME = "file_processor.db"
DB_FILE = os.path.join(os.getgwd() , "database_files" , os.getenv("DB_SERVER_FILE", DEFAULT_DB_FILENAME))
HOST = os.getenv("DB_SERVER_HOST", "127.0.0.1")
PORT = int(os.environ.get("DB_SERVER_PORT", 8034)) # Use a distinct port

app = Flask(__name__)

# --- Initialization (ensure init_db is called, which now handles schema updates) ---
print(f"Checking/Initializing database: '{DB_FILE}'...")
if not database.init_db(DB_FILE): # init_db now handles adding columns
    print(f"FATAL: Failed to initialize/update database '{DB_FILE}'. Exiting.")
    sys.exit(1)
else:
    print(f"Database '{DB_FILE}' ready.")


# --- API Routes ---

@app.route('/status', methods=['GET'])
def get_status():
    """Simple status check endpoint."""
    return jsonify({"status": "ok", "db_file": os.path.abspath(DB_FILE)}), 200

@app.route('/add_entry', methods=['POST'])
def add_entry():
    """Adds a new log entry, accepting optional original/sanitized tags."""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    metadata = data.get('metadata')
    github_info = data.get('github_info') # Can be None or contain tags

    if not metadata:
        return jsonify({"error": "Missing 'metadata' in request body"}), 400

    # --- MODIFICATION: Pass full github_info (which might contain new tags) ---
    # database.add_log_entry is already updated to handle extracting the tags from github_info
    entry_id = database.add_log_entry(DB_FILE, metadata, github_info)
    # --- END MODIFICATION ---

    if entry_id is not None:
        return jsonify({"message": "Entry added successfully", "id": entry_id}), 201 # 201 Created
    else:
        # Log entry function already logged the specific error
        return jsonify({"error": "Failed to add entry to database (check server logs)"}), 500

# --- MODIFICATION: Query routes now return the new tag fields via updated DB functions ---
@app.route('/entries', methods=['GET'])
def get_entries():
    """Gets a summary list of *active* entries (includes new tag fields)."""
    entries = database.get_all_entries(DB_FILE) 
    return jsonify(entries), 200

@app.route('/entries/trash', methods=['GET'])
def get_trash_entries():
    """Gets a summary list of *trashed* entries (includes new tag fields)."""
    entries = database.get_trash_entries(DB_FILE)
    return jsonify(entries), 200

@app.route('/entry/<int:entry_id>', methods=['GET'])
def get_entry(entry_id):
    """Gets full details for a specific *active* entry (includes new tag fields)."""
    details = database.get_entry_details(DB_FILE, entry_id, from_trash=False)
    if details:
        return jsonify(details), 200
    else:
        return jsonify({"error": f"Active entry with ID {entry_id} not found or failed to load"}), 404

@app.route('/entry/trash/<int:entry_id>', methods=['GET'])
def get_trash_entry(entry_id):
    """Gets full details for a specific *trashed* entry (includes new tag fields)."""
    details = database.get_entry_details(DB_FILE, entry_id, from_trash=True)
    if details:
        return jsonify(details), 200
    else:
        return jsonify({"error": f"Trashed entry with ID {entry_id} not found or failed to load"}), 404
# --- END MODIFICATION ---
# --- Add this route inside db_server.py ---

# Import necessary functions/modules if not already present at the top
from flask import request, jsonify 
# import database # Make sure database module is imported

@app.route('/entry/<int:entry_id>/set_public', methods=['POST'])
def set_public(entry_id):
    """Sets the public status for a given entry."""
    # Ensure the request body is JSON
    if not request.is_json:
        database.print_db_log("WARN", f"/set_public called for ID {entry_id} without JSON body.")
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    is_public_val = data.get('public') # Expecting {"public": true} or {"public": false}

    # Validate the input value
    if is_public_val is None or not isinstance(is_public_val, bool):
         database.print_db_log("WARN", f"/set_public called for ID {entry_id} with invalid 'public' field: {data}")
         return jsonify({"error": "Missing or invalid 'public' boolean field (true/false) in request body"}), 400

    database.print_db_log("INFO", f"Received request via /set_public for ID {entry_id} to set public={is_public_val}")
    
    # Call the database function (ensure database.py has set_public_status)
    # Make sure DB_FILE is defined globally in db_server.py or passed correctly
    success = database.set_public_status(DB_FILE, entry_id, is_public_val) 

    if success:
        status_str = "public" if is_public_val else "private"
        database.print_db_log("INFO", f"Successfully set public status for ID {entry_id} via API.")
        return jsonify({"message": f"Entry ID {entry_id} status set to {status_str}"}), 200
    else:
         # The database function logs details if ID not found or DB error
         database.print_db_log("ERROR", f"Failed to set public status for ID {entry_id} via API (check DB logs).")
         # Return 404 if the entry likely didn't exist for update
         return jsonify({"error": f"Failed to update public status for entry ID {entry_id} (entry not found or database error)"}), 404 
# --- End of route to add ---
@app.route('/delete_entry/<int:entry_id>', methods=['DELETE'])
def delete_entry(entry_id):
    """Moves an entry from 'processed_files' to 'trash'."""
    # --- No change needed here, DB function handles moving all fields ---
    success = database.move_to_trash(DB_FILE, entry_id)
    if success:
        return jsonify({"message": f"Entry ID {entry_id} moved to trash successfully"}), 200
    else:
        return jsonify({"error": f"Failed to move entry ID {entry_id} to trash (may not exist or DB error)"}), 404 


# --- Run Server (remains the same) ---
if __name__ == '__main__':
    print(f"Starting DB Server on http://{HOST}:{PORT}...")
    print(f"Using database: {os.path.abspath(DB_FILE)}")
    # Use host='0.0.0.0' to make it accessible externally if needed
    # Use debug=False for production/stable use
    app.run(host=HOST, port=PORT, debug=False)