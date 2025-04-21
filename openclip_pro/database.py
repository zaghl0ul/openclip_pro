import os
import sqlite3
import json
import logging
import time
import uuid
from typing import Dict, List, Optional

from config import DB_FILE

# Set up logger for this module
logger = logging.getLogger(__name__)

# --- Database Connection Context Manager ---

# Using a context manager simplifies connection handling and ensures closure.
# However, for functions performing multiple operations that need to be atomic
# (like save_project, delete_project), we need explicit transaction management
# within the function using conn.commit() and conn.rollback().

# --- Database Setup ---


def setup_database():
    """
    Initialize the SQLite database.

    Creates the necessary tables (settings, projects, clips) if they don't exist
    and initializes default settings.
    """
    conn = None
    try:
        # Ensure the directory for the database file exists
        db_dir = os.path.dirname(DB_FILE)
        if db_dir: # Avoid trying to create empty string directory
            os.makedirs(db_dir, exist_ok=True)

        # Connect to the database
        conn = sqlite3.connect(DB_FILE, timeout=10) # Added timeout
        cursor = conn.cursor()

        # Use WAL mode for better concurrency (optional, but often recommended)
        cursor.execute("PRAGMA journal_mode=WAL;")

        # Enable foreign key constraints
        cursor.execute("PRAGMA foreign_keys = ON;")

        # Create settings table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )

        # Create projects table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL, -- ISO 8601 format recommended
                source_type TEXT NOT NULL,
                source_path TEXT NOT NULL,
                base_dir_path TEXT NOT NULL,
                -- Store ID of the highest-scoring clip for quick preview/reference
                top_clip_id TEXT,
                -- Store additional non-indexed project metadata as JSON
                data JSON,
                -- Foreign key constraint to ensure top_clip_id refers to a valid clip
                -- Deferrable initially deferred allows inserting project before top clip exists
                FOREIGN KEY (top_clip_id) REFERENCES clips(id) ON DELETE SET NULL DEFERRABLE INITIALLY DEFERRED
            )
            """
        )

        # Create clips table
        # Separated from project JSON data for better query performance on core fields.
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS clips (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                start REAL NOT NULL, -- Start time in seconds
                end REAL NOT NULL,   -- End time in seconds
                score INTEGER NOT NULL, -- Score assigned by analysis
                tag TEXT,           -- Optional tag (e.g., 'goal', 'funny')
                category TEXT,      -- Optional category
                clip_path TEXT,     -- Path to the generated video file
                thumbnail TEXT,     -- Path to the thumbnail image file
                -- Store full clip data including AI analysis details, etc.
                data JSON,
                -- Ensures clips are deleted when their parent project is deleted
                FOREIGN KEY (project_id) REFERENCES projects (id) ON DELETE CASCADE
            )
            """
        )
        # Add indexes for frequently queried columns
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_clips_project_id ON clips (project_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_clips_score ON clips (score)")


        # Add default settings if they don't exist
        _initialize_default_settings(cursor)

        conn.commit()
        logger.info("Database setup or verification completed successfully.")
        return True

    except sqlite3.Error as e:
        logger.error(f"SQLite error during database setup: {e}", exc_info=True)
        if conn:
            conn.rollback() # Rollback changes if error occurs
        # Reraise the exception to signal failure
        raise RuntimeError(f"Database setup failed: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error during database setup: {e}", exc_info=True)
        if conn:
            conn.rollback()
        raise RuntimeError(f"Database setup failed unexpectedly: {e}") from e
    finally:
        if conn:
            conn.close()


def _initialize_default_settings(cursor: sqlite3.Cursor):
    """
    Insert default application settings into the settings table if they are not present.

    Args:
        cursor: The database cursor object.
    """
    default_settings = {
        "app_version": "1.1",
        "default_project_name": "Project %Y-%m-%d",
        "default_clip_length": "60",        # Default duration for clips in seconds
        "default_frame_sample_rate": "2.5", # Frames per second to sample for analysis
        "default_score_threshold": "75",    # Minimum score for a clip to be considered 'good'
        "default_ai_provider": "openai",    # AI service provider
        "default_ai_model": "gpt-4o",       # Specific AI model
        "export_format": "web_optimized",   # Default video export format
        "compression_quality": "85",        # Export compression quality (e.g., 0-100)
        "max_resolution": "720",            # Maximum export resolution height (e.g., 720p)
        "max_workers_extraction": "4",      # Max parallel workers for frame extraction
        "max_workers_encoding": "4",        # Max parallel workers for video encoding
        "max_workers_api": "8",             # Max parallel workers for API calls (AI analysis)
        "max_workers_clip_gen": "4",        # Max parallel workers for clip generation
        "default_theme": "dark",            # UI theme preference
    }

    added_count = 0
    # Use INSERT OR IGNORE to add only missing default settings
    for key, value in default_settings.items():
        try:
            # Ensure value is stored as string (TEXT column)
            cursor.execute(
                "INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)",
                (key, str(value)),
            )
            if cursor.rowcount > 0:
                added_count += 1
        except sqlite3.Error as e:
            # Log specific error but continue trying other settings
            logger.error(f"Error inserting default setting '{key}': {e}", exc_info=True)

    if added_count > 0:
        logger.info(f"Initialized {added_count} default settings.")


# --- Settings Management ---

def get_settings() -> Dict[str, str]:
    """
    Retrieve all application settings from the database.

    Returns:
        A dictionary containing setting keys and their values.
        Returns an empty dictionary if an error occurs.
    """
    settings = {}
    try:
        # Use context manager for connection handling
        with sqlite3.connect(DB_FILE, timeout=10) as conn:
            conn.row_factory = sqlite3.Row # Access columns by name
            cursor = conn.cursor()
            cursor.execute("SELECT key, value FROM settings")
            # Convert list of Row objects to a dictionary
            settings = {row["key"]: row["value"] for row in cursor.fetchall()}
        logger.debug(f"Fetched {len(settings)} settings.")
    except sqlite3.Error as e:
        logger.error(f"Error getting settings: {e}", exc_info=True)
        # Return empty dict on error
        return {}
    except Exception as e:
        logger.error(f"Unexpected error getting settings: {e}", exc_info=True)
        # Return empty dict on error
        return {}
    return settings


def save_setting(key: str, value: any) -> bool:
    """
    Save a single setting to the database. Updates if key exists, inserts otherwise.

    Args:
        key: The name of the setting.
        value: The value of the setting (will be converted to string).

    Returns:
        True if the setting was saved or updated, False otherwise.
    """
    try:
        # Use context manager for connection handling and implicit commit/rollback on exit
        with sqlite3.connect(DB_FILE, timeout=10) as conn:
            cursor = conn.cursor()

            # Check current value to avoid unnecessary writes (optional optimization)
            cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
            existing = cursor.fetchone()
            str_value = str(value) # Ensure value is compared and stored as string

            if existing and existing[0] == str_value:
                logger.debug(f"Setting '{key}' value unchanged ('{str_value}'), no update needed.")
                return True # Indicate success even if no change occurred

            # INSERT OR REPLACE atomically handles insert or update
            cursor.execute(
                "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
                (key, str_value),
            )
            # Context manager handles commit on successful exit
            logger.info(f"Setting '{key}' saved with value '{str_value}'.")
            return True
    except sqlite3.Error as e:
        logger.error(f"Error saving setting '{key}': {e}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"Unexpected error saving setting '{key}': {e}", exc_info=True)
        return False


# --- Project and Clip Management ---

def save_project(project_data: Dict) -> bool:
    """
    Save or update a project and its associated clips in the database.

    This function handles the entire project structure. If 'clips' are present
    in `project_data`, it assumes this is the complete set of clips for the
    project and will REPLACE all existing clips for that project_id.

    Args:
        project_data: A dictionary containing project details and optionally a 'clips' list.
                      Must include an 'id' key.

    Returns:
        True if the project and clips were saved successfully, False otherwise.
    """
    project_id = project_data.get("id")
    if not project_id:
        logger.error("Cannot save project: Missing required 'id' field.")
        return False

    conn = None
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10)
        # Enable foreign keys for this connection
        conn.execute("PRAGMA foreign_keys = ON;")

        # Begin transaction for atomic save of project and clips
        cursor = conn.cursor()
        cursor.execute("BEGIN")

        # Extract clips from the input data. Use pop to remove it from the data
        # that will be stored in the project's JSON 'data' column.
        # Default to None if 'clips' key is not present.
        clips_to_save = project_data.pop("clips", None)

        # Determine the top clip ID if clips are being saved
        top_clip_id = None
        if clips_to_save: # Only calculate if clips are provided
            # Find the clip with the maximum score. Returns None if clips_to_save is empty.
            top_clip = max(clips_to_save, key=lambda c: c.get("score", 0), default=None)
            if top_clip:
                # Ensure the top clip has an ID (should be assigned earlier)
                top_clip_id = top_clip.get("id")
                if not top_clip_id:
                   logger.warning(f"Top clip for project {project_id} is missing an ID during save.")


        # Prepare project record data for insertion/update
        # Default values prevent errors if keys are missing in project_data
        project_record = {
            "id": project_id,
            "name": project_data.get("name", "Untitled Project"),
            "created_at": project_data.get(
                "created_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) # Use UTC ISO 8601
            ),
            "source_type": project_data.get("source_type", "unknown"),
            "source_path": project_data.get("source_path", ""),
            "base_dir_path": project_data.get("base_dir_path", ""),
            "top_clip_id": top_clip_id,
            # Store the rest of the project_data (without clips) as JSON
            "data": json.dumps(project_data),
        }

        # Use INSERT OR REPLACE to simplify insert/update logic for the project
        cursor.execute(
            """
            INSERT OR REPLACE INTO projects
                (id, name, created_at, source_type, source_path, base_dir_path, top_clip_id, data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                project_record["id"],
                project_record["name"],
                project_record["created_at"],
                project_record["source_type"],
                project_record["source_path"],
                project_record["base_dir_path"],
                project_record["top_clip_id"],
                project_record["data"],
            ),
        )
        logger.info(f"Saved/Updated project record {project_id}.")

        # If clips were provided, replace existing clips for this project
        if clips_to_save is not None: # Check for None explicitly to allow saving empty list
            # 1. Delete existing clips for this project
            cursor.execute("DELETE FROM clips WHERE project_id = ?", (project_id,))
            logger.debug(f"Removed existing clips for project {project_id} before saving new set.")

            # 2. Prepare and insert the new set of clips
            clip_data_to_insert = []
            for clip in clips_to_save:
                # Ensure each clip has a unique ID; generate if missing.
                clip_id = clip.get("id") or str(uuid.uuid4())
                clip["id"] = clip_id  # Ensure the ID exists in the JSON data as well

                clip_data_to_insert.append(
                    (
                        clip_id,
                        project_id,
                        clip.get("start", 0.0), # Use float for REAL type
                        clip.get("end", 0.0),   # Use float for REAL type
                        clip.get("score", 0),
                        clip.get("tag"),        # Allow None
                        clip.get("category"),   # Allow None
                        clip.get("clip_path"),  # Allow None
                        clip.get("thumbnail"),  # Allow None
                        json.dumps(clip),       # Store the full clip dict as JSON
                    )
                )

            # Use executemany for efficient bulk insertion
            if clip_data_to_insert:
                cursor.executemany(
                    """
                    INSERT INTO clips
                        (id, project_id, start, end, score, tag, category, clip_path, thumbnail, data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    clip_data_to_insert,
                )
                logger.info(f"Inserted {len(clip_data_to_insert)} clips for project {project_id}.")
            else:
                 logger.info(f"No clips provided or clip list was empty for project {project_id}. All existing clips removed.")


        # If we reached here without errors, commit the transaction
        conn.commit()
        logger.info(f"Successfully saved project {project_id} and its clips.")
        return True

    except sqlite3.Error as e:
        logger.error(f"SQLite error saving project {project_id}: {e}", exc_info=True)
        if conn:
            conn.rollback() # Rollback on error
        return False
    except Exception as e:
        logger.error(f"Unexpected error saving project {project_id}: {e}", exc_info=True)
        if conn:
            conn.rollback() # Rollback on error
        return False
    finally:
        if conn:
            conn.close()


def update_clip_data(clip_id: str, updates: Dict) -> bool:
    """
    Update specific fields of a single clip.

    Updates both the core indexed columns (score, tag, category) if present
    in `updates`, and merges the `updates` dictionary into the clip's JSON `data` column.

    Args:
        clip_id: The ID of the clip to update.
        updates: A dictionary containing the fields and new values to update.

    Returns:
        True if the clip was updated successfully, False otherwise.
    """
    conn = None
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10)
        # Enable foreign keys for this connection (though less critical for update)
        conn.execute("PRAGMA foreign_keys = ON;")
        cursor = conn.cursor()

        # Begin transaction for atomic update of core columns and JSON data
        cursor.execute("BEGIN")

        # 1. Fetch the current JSON data
        cursor.execute("SELECT data FROM clips WHERE id = ?", (clip_id,))
        result = cursor.fetchone()
        if not result:
            logger.warning(f"Clip {clip_id} not found for update.")
            # No rollback needed as no changes were made yet
            return False

        try:
            current_data = json.loads(result[0])
        except (json.JSONDecodeError, TypeError) as json_err:
            logger.error(f"Failed to decode existing JSON data for clip {clip_id}: {json_err}", exc_info=True)
            # No rollback needed yet
            return False

        # 2. Merge updates into the current data
        current_data.update(updates)
        updated_json_data = json.dumps(current_data)

        # 3. Update the JSON data column
        cursor.execute(
            "UPDATE clips SET data = ? WHERE id = ?",
            (updated_json_data, clip_id),
        )

        # 4. Update core columns if they are present in the updates dictionary
        #    This keeps indexed columns synchronized with the JSON data for performance.
        core_fields_to_update = {}
        if "score" in updates:
            core_fields_to_update["score"] = updates["score"]
        if "tag" in updates:
            core_fields_to_update["tag"] = updates["tag"]
        if "category" in updates:
            core_fields_to_update["category"] = updates["category"]
        if "start" in updates:
            core_fields_to_update["start"] = updates["start"]
        if "end" in updates:
            core_fields_to_update["end"] = updates["end"]
        if "clip_path" in updates:
            core_fields_to_update["clip_path"] = updates["clip_path"]
        if "thumbnail" in updates:
            core_fields_to_update["thumbnail"] = updates["thumbnail"]

        if core_fields_to_update:
            set_clause = ", ".join([f"{k} = ?" for k in core_fields_to_update])
            values = list(core_fields_to_update.values()) + [clip_id]
            update_sql = f"UPDATE clips SET {set_clause} WHERE id = ?"
            cursor.execute(update_sql, values)
            logger.debug(f"Updated core columns for clip {clip_id}: {', '.join(core_fields_to_update.keys())}")

        # Commit the transaction if all updates were successful
        conn.commit()
        logger.info(f"Successfully updated clip {clip_id}.")
        return True

    except (sqlite3.Error, json.JSONDecodeError) as e:
        logger.error(f"Error updating clip {clip_id}: {e}", exc_info=True)
        if conn:
            conn.rollback() # Rollback on error
        return False
    except Exception as e:
        logger.error(f"Unexpected error updating clip {clip_id}: {e}", exc_info=True)
        if conn:
            conn.rollback() # Rollback on error
        return False
    finally:
        if conn:
            conn.close()


# Alias for semantic clarity if AI analysis primarily updates clip data.
# Ensures consistency if the update mechanism changes later.
save_ai_analysis_to_clip = update_clip_data


def load_projects() -> List[Dict]:
    """
    Load a summary list of all projects.

    Includes basic project metadata, the count of associated clips, and
    thumbnail/score from the project's designated 'top clip'.

    Returns:
        A list of dictionaries, each representing a project summary.
        Returns an empty list if an error occurs.
    """
    projects = []
    try:
        # Use context manager for connection
        with sqlite3.connect(DB_FILE, timeout=10) as conn:
            conn.row_factory = sqlite3.Row # Access columns by name
            cursor = conn.cursor()

            # Query to get project info, clip count, and top clip details
            # LEFT JOIN ensures projects without clips or without a top clip are still listed.
            # The top clip info is fetched by joining based on `projects.top_clip_id`.
            cursor.execute(
                """
                SELECT
                    p.id,
                    p.name,
                    p.created_at,
                    p.source_type,
                    p.source_path,
                    p.base_dir_path,
                    COUNT(c.id) as clip_count, -- Count clips associated with the project
                    tc.thumbnail as top_clip_thumbnail, -- Get thumbnail from the top clip
                    tc.score as top_clip_score          -- Get score from the top clip
                FROM projects p
                -- Left join to count all clips for the project
                LEFT JOIN clips c ON p.id = c.project_id
                -- Left join to get details of the specific top clip referenced by project
                LEFT JOIN clips tc ON p.top_clip_id = tc.id
                GROUP BY p.id -- Group by project to count clips correctly
                ORDER BY p.created_at DESC -- Show newest projects first
                """
            )

            projects = [dict(row) for row in cursor.fetchall()]
            logger.info(f"Loaded {len(projects)} project summaries.")

    except sqlite3.Error as e:
        logger.error(f"Error loading project summaries: {e}", exc_info=True)
        return [] # Return empty list on error
    except Exception as e:
        logger.error(f"Unexpected error loading project summaries: {e}", exc_info=True)
        return [] # Return empty list on error

    return projects


def load_project(project_id: str) -> Optional[Dict]:
    """
    Load a full project, including its details and all associated clips.

    Retrieves project data primarily from the JSON 'data' column in the 'projects' table,
    and fetches all associated clips, reconstructing the full project dictionary.

    Args:
        project_id: The ID of the project to load.

    Returns:
        A dictionary containing the full project data (including clips),
        or None if the project is not found or an error occurs.
    """
    try:
        # Use context manager for connection
        with sqlite3.connect(DB_FILE, timeout=10) as conn:
            conn.row_factory = sqlite3.Row # Access columns by name
            cursor = conn.cursor()

            # 1. Get the base project data stored in the JSON column
            cursor.execute(
                "SELECT id, name, data FROM projects WHERE id = ?", (project_id,)
            )
            project_row = cursor.fetchone()

            if not project_row:
                logger.warning(f"Project {project_id} not found in database.")
                return None

            # Load project data from JSON, fallback for potential decoding issues
            try:
                # Use project_row['data'] which might be None or valid JSON string
                project_data = json.loads(project_row["data"] or '{}')
            except (json.JSONDecodeError, TypeError) as json_err:
                 logger.error(f"Failed to decode project JSON data for {project_id}: {json_err}. "
                              f"Falling back to basic info.", exc_info=True)
                 project_data = {} # Start with an empty dict if JSON is bad

            # Ensure essential fields (like id, name) are present, using DB columns as authoritative source
            project_data["id"] = project_row["id"]
            project_data["name"] = project_row["name"] # Overwrite or add name from core column

            # 2. Get all associated clips, loading their full data from the JSON column
            cursor.execute(
                """
                SELECT data
                FROM clips
                WHERE project_id = ?
                ORDER BY score DESC, start ASC -- Order clips meaningfully
                """,
                (project_id,),
            )

            clips = []
            for clip_row in cursor.fetchall():
                try:
                    # Ensure clip_row['data'] is not None before loading
                    if clip_row['data']:
                         clips.append(json.loads(clip_row["data"]))
                    else:
                        logger.warning(f"Clip within project {project_id} has NULL data column.")
                except (json.JSONDecodeError, TypeError) as json_err:
                    logger.error(f"Failed to decode clip JSON data within project {project_id}: {json_err}", exc_info=True)
                    # Optionally skip the problematic clip or add placeholder

            project_data["clips"] = clips
            logger.info(f"Loaded project {project_id} with {len(clips)} clips.")
            return project_data

    except sqlite3.Error as e:
        logger.error(f"Error loading project {project_id}: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading project {project_id}: {e}", exc_info=True)
        return None


def delete_project(project_id: str) -> bool:
    """
    Delete a project and all its associated clips from the database.

    Relies on the `ON DELETE CASCADE` foreign key constraint on the `clips` table
    to automatically remove clips when the parent project is deleted.

    Args:
        project_id: The ID of the project to delete.

    Returns:
        True if the project was deleted successfully, False otherwise (e.g., not found).
    """
    conn = None
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10)
        # Enable foreign keys to ensure CASCADE works
        conn.execute("PRAGMA foreign_keys = ON;")
        cursor = conn.cursor()

        # Begin transaction
        cursor.execute("BEGIN")

        # Delete the project. Associated clips will be deleted automatically
        # due to the 'ON DELETE CASCADE' constraint in the clips table schema.
        cursor.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        rows_affected = cursor.rowcount

        # Commit the transaction
        conn.commit()

        if rows_affected > 0:
            logger.info(f"Successfully deleted project {project_id} and its associated clips (via cascade).")
            return True
        else:
            logger.warning(f"Project {project_id} not found for deletion.")
            return False

    except sqlite3.Error as e:
        logger.error(f"Error deleting project {project_id}: {e}", exc_info=True)
        if conn:
            conn.rollback() # Rollback on error
        return False
    except Exception as e:
        logger.error(f"Unexpected error deleting project {project_id}: {e}", exc_info=True)
        if conn:
            conn.rollback() # Rollback on error
        return False
    finally:
        if conn:
            conn.close()

# Example Usage (Optional - for testing)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    logger.info("Running database operations example...")

    # Ensure database is setup
    try:
        setup_database()
    except RuntimeError as e:
        logger.error(f"Exiting due to database setup failure: {e}")
        exit(1)


    # Settings example
    save_setting("default_theme", "light")
    save_setting("new_setting", 123.45)
    settings = get_settings()
    logger.info(f"Current Settings: {settings}")

    # Project example
    proj_id = str(uuid.uuid4())
    clip1_id = str(uuid.uuid4())
    clip2_id = str(uuid.uuid4())

    new_project = {
        "id": proj_id,
        "name": "Test Project Alpha",
        "source_type": "video_file",
        "source_path": "/path/to/video.mp4",
        "base_dir_path": "/path/to/project/files",
        "some_other_data": {"info": "value1", "nested": True},
        "clips": [
            {
                "id": clip1_id, "start": 10.5, "end": 25.0, "score": 85,
                "tag": "highlight", "category": "gameplay",
                "clip_path": f"/path/to/project/files/clips/{clip1_id}.mp4",
                "thumbnail": f"/path/to/project/files/thumbs/{clip1_id}.jpg",
                "analysis": {"details": "some ai output 1"}
            },
            {
                "id": clip2_id, "start": 50.0, "end": 62.3, "score": 92, # Higher score
                "tag": "goal", "category": "sports",
                "clip_path": f"/path/to/project/files/clips/{clip2_id}.mp4",
                "thumbnail": f"/path/to/project/files/thumbs/{clip2_id}.jpg",
                "analysis": {"details": "some ai output 2"}
            }
        ]
    }

    if save_project(new_project):
        logger.info("Project saved successfully.")

        # Load summaries
        project_list = load_projects()
        logger.info(f"Project Summaries: {json.dumps(project_list, indent=2)}")

        # Load full project
        loaded_project = load_project(proj_id)
        if loaded_project:
            logger.info(f"Loaded Project: {json.dumps(loaded_project, indent=2)}")
            # Verify top clip details match project summary
            top_clip_in_summary = next((p for p in project_list if p['id'] == proj_id), None)
            if top_clip_in_summary:
                logger.info(f"Summary top clip score: {top_clip_in_summary.get('top_clip_score')}, "
                            f"thumbnail: {top_clip_in_summary.get('top_clip_thumbnail')}")


            # Update a clip
            update_clip_data(clip1_id, {"score": 88, "tag": "updated_highlight", "new_field": "added"})
            updated_project = load_project(proj_id)
            if updated_project:
                 logger.info(f"Updated Project: {json.dumps(updated_project, indent=2)}")

        # Delete project
        # time.sleep(1) # Optional pause
        # if delete_project(proj_id):
        #     logger.info("Project deleted successfully.")
        #     project_list_after_delete = load_projects()
        #     logger.info(f"Project Summaries after delete: {json.dumps(project_list_after_delete, indent=2)}")
        # else:
        #      logger.error("Project deletion failed.")

    else:
        logger.error("Failed to save project.")
