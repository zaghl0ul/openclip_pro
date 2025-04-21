import streamlit as st
import sys
import os
import subprocess

# Add the project root to sys.path
# Assumes script is run from project root or APP_DIR is correct
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.append(APP_DIR)

import logging
import time

# --- Configuration and Setup ---
from config import LOGGING_CONFIG, APP_DIR, DEFAULT_TEMP_DIR  # Import DEFAULT_TEMP_DIR
from database import setup_database, get_settings

# Import the main AI module initializer and UI components
from ai import initialize_ai_integration
from ai.ui_components import (
    create_api_key_management_ui,
)  # Import only the API key management UI

# --- Setup Logging ---
logging.basicConfig(
    level=LOGGING_CONFIG["level"],
    format=LOGGING_CONFIG["format"],
    datefmt=LOGGING_CONFIG["datefmt"],
)
logger = logging.getLogger(__name__)
logger.info("Application starting...")
logger.info(f"APP_DIR: {APP_DIR}")
logger.info(f"DEFAULT_TEMP_DIR: {DEFAULT_TEMP_DIR}")

# --- Database Setup ---
try:
    setup_database()
    logger.info("Database setup complete.")
except Exception as db_err:
    logger.error(f"Database setup failed: {db_err}", exc_info=True)
    st.error(
        f"Fatal Error: Could not initialize database ({db_err}). Please check permissions and restart."
    )
    st.stop()  # Stop execution if DB fails

# --- Load Settings ---
# Store settings in session state to avoid repeated DB calls per interaction
if "user_settings" not in st.session_state:
    st.session_state.user_settings = get_settings()
    logger.info("Loaded application settings.")

# --- Initialize AI Module ---
# This now uses the function from ai/__init__.py which handles caching in session state
ai_module = initialize_ai_integration()
# The initialize_ai_integration function is designed to cache in session state itself,
# but this explicit check/assignment doesn't hurt and ensures 'ai_module' is always set.
if "ai_module" not in st.session_state:
    st.session_state.ai_module = ai_module

# --- Initialize Model Registry ---
# Model registry should be part of the AI module instance
if ai_module and not hasattr(st.session_state, "model_registry"):
    st.session_state.model_registry = (
        ai_module.model_registry
    )  # Access registry via module

# --- Initialize Session State Defaults ---
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "🏠 Home"
if "current_project_id" not in st.session_state:
    st.session_state.current_project_id = None
if "current_view" not in st.session_state:  # For project list view (list/gallery)
    st.session_state.current_view = "list"
if "clip_ai_analysis" not in st.session_state:  # Cache for AI board results per clip
    st.session_state.clip_ai_analysis = {}
# Add other specific session state keys as needed (e.g., for confirmations)
if "confirm_delete_home" not in st.session_state:
    st.session_state.confirm_delete_home = None
if "confirm_delete_list" not in st.session_state:
    st.session_state.confirm_delete_list = None
if "confirm_delete_gallery" not in st.session_state:
    st.session_state.confirm_delete_gallery = None
if "confirming_delete_project" not in st.session_state:
    st.session_state.confirming_delete_project = None

# --- Streamlit Page Config ---
# Apply theme from settings if available
current_theme = st.session_state.user_settings.get("default_theme", "dark")
st.set_page_config(
    page_title="OpenClip Pro - Visual Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
    # Note: theme is not a valid parameter for set_page_config
    # It's controlled through config.toml or streamlit theme settings
)

# --- UI Imports ---
# Import page functions from the ui package
from ui import home_page, projects_list_page
from ui.create_project_page import create_project_page  # <- direct import instead
    # api_keys_page # REMOVED - now uses ui_components
)


# Create a dedicated page function for API keys
def api_keys_display_page():
    """Displays the API Key management UI using AI components."""
    # Get the AI module instance
    ai_module_instance = st.session_state.get("ai_module")
    if not ai_module_instance:
        st.error("AI Module is not available. Cannot manage API keys.")
        return

    # Use the UI component function from ai.ui_components
    create_api_key_management_ui(
        ai_module_instance.key_manager, ai_module_instance.model_registry
    )


# --- Sidebar Navigation ---
def sidebar_navigation():
    with st.sidebar:
        st.title("🎬 OpenClip Pro")
        st.caption(
            f"v{st.session_state.user_settings.get('app_version', 'N/A')} | AI Enhanced"
        )  # Use settings for version

        st.header("Navigation")
        # Define modes
        base_modes = ["🏠 Home", "🎬 Create New", "📁 Projects"]
        # Add AI-specific modes if AI module is available and supports them
        ai_modes = []
        if st.session_state.get("ai_module"):
            ai_modes.append("🔑 API Keys")

        config_mode = ["⚙️ Settings"]

        all_modes = base_modes + ai_modes + config_mode

        # Use radio for selection
        # Get current mode safely
        current_mode = st.session_state.get("app_mode", "🏠 Home")
        if current_mode not in all_modes:
            current_mode = "🏠 Home"  # Fallback
        current_index = all_modes.index(current_mode)

        # Use a unique key and on_change for mode switching
        selected_mode = st.radio(
            "Main Menu",
            all_modes,
            index=current_index,
            key="main_menu_radio",
            # on_change callback is simpler than checking button clicks if logic is simple
            # on_change=lambda: st.session_state.update({"app_mode": st.session_state.main_menu_radio})
        )

        # Update app_mode if radio button changed it
        if selected_mode != st.session_state.app_mode:
            logger.info(
                f"Navigating from {st.session_state.app_mode} to {selected_mode}"
            )
            st.session_state.app_mode = selected_mode
            # Clear project ID when navigating away from project details
            if selected_mode != "📁 Projects":
                st.session_state.current_project_id = None
            # Clear confirmations when navigating away from pages that use them
            st.session_state.confirm_delete_home = None
            st.session_state.confirm_delete_list = None
            st.session_state.confirm_delete_gallery = None
            st.session_state.confirming_delete_project = None
            # Clear cached AI board analysis results if navigating away from project details
            # This might be too aggressive, maybe clear on project change instead?
            # st.session_state.clip_ai_analysis = {}
            st.rerun()

        # No AI Configuration UI in sidebar

        st.divider()
        st.info(
            "Need Help? [Docs Placeholder](https://example.com)", icon="❓"
        )  # TODO: Replace with real docs link
        st.markdown("---")
        st.caption("© 2024 OpenClip Pro")


# --- Main Application Router ---
def main():
    # AI Model Registry is now accessed via st.session_state.ai_module.model_registry
    # Initialize AI models/providers in the sidebar UI component

    sidebar_navigation()

    active_mode = st.session_state.get("app_mode", "🏠 Home")
    current_project_id = st.session_state.get("current_project_id")

    # Route to the appropriate page function
    if active_mode == "🏠 Home":
        home_page.show()
    elif active_mode == "🎬 Create New":
        # Pass necessary data explicitly or rely on session state
        create_project_page.show()
    elif active_mode == "📁 Projects":
        if current_project_id:
            project_details_page.show(current_project_id)  # Passes project_id
        else:
            projects_list_page.show()
    elif active_mode == "⚙️ Settings":
        settings_page.show()
    elif active_mode == "🔑 API Keys":
        # Check if ai_module is available before showing the page
        if st.session_state.get("ai_module"):
            api_keys_display_page()  # Call the dedicated display function
        else:
            st.error("AI Module not available. Cannot access API Key settings.")
            st.button(
                "Go Home",
                on_click=lambda: st.session_state.update({"app_mode": "🏠 Home"}),
            )

    else:
        st.error(f"Invalid application mode: {active_mode}. Returning home.")
        st.session_state.app_mode = "🏠 Home"
        time.sleep(1)
        st.rerun()


if __name__ == "__main__":
    # Check for ffmpeg/ffprobe (basic check)
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
        subprocess.run(["ffprobe", "-version"], check=True, capture_output=True)
        logger.info("ffmpeg and ffprobe found in PATH.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        st.error(
            "Fatal Error: ffmpeg and/or ffprobe not found. These are required dependencies."
        )
        st.info("Please install ffmpeg and ensure it's in your system's PATH.")
        logger.error("ffmpeg or ffprobe not found.")
        sys.exit("ffmpeg or ffprobe not found.")  # Exit application

    main()

    # Cleanup function registration - still unreliable with Streamlit's execution model.
    # A better approach for resource cleanup might be to rely on session state
    # or Streamlit component lifecycle if available in future versions.
    # def app_cleanup():
    #     if 'ai_module' in st.session_state and st.session_state.ai_module:
    #         logger.info("Running app cleanup...")
    #         # Assuming AIAnalysisModule has a close/cleanup method for clients/sessions
    #         import asyncio
    #         try:
    #              asyncio.run(st.session_state.ai_module.close())
    #              logger.info("AI module cleanup finished.")
    #         except Exception as e:
    #             logger.error(f"Error during AI module cleanup: {e}", exc_info=True)
    #     # Other cleanup (e.g., temp files not in project dirs?)
    # try:
    #      import atexit
    #      # Registering async cleanup with sync atexit is tricky.
    #      # This is left as a placeholder for future improvement.
    #      # atexit.register(lambda: asyncio.run(app_cleanup()))
    #      pass
    # except Exception as e:
    #      logger.warning(f"Could not register cleanup function: {e}")
