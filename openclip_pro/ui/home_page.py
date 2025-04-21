
import streamlit as st
import datetime
import os
import logging

# Import database and media utility functions
from database import load_projects, delete_project
from media_utils import cleanup_project_directories

# Setup logger for this module
logger = logging.getLogger(__name__)

# --- Constants ---
MAX_RECENT_PROJECTS = 3  # Maximum number of recent projects to display on the home page
CONFIRM_DELETE_STATE_KEY = "confirm_delete_home"  # Session state key for delete confirmation


def _display_delete_confirmation(project_info: dict):
    """Displays the confirmation dialog for project deletion."""
    pid = project_info["id"]
    pname = project_info["name"]
    pdir = project_info["dir"]

    st.warning(f"⚠️ Confirm permanent deletion of project '{pname}'?")
    st.caption(
        "This will delete the database entry and attempt to remove associated files."
    )
    confirm_cols = st.columns(2)

    # --- Confirmation Buttons ---
    with confirm_cols[0]:
        if st.button("✅ Yes, Delete Permanently", key=f"confirm_del_yes_{pid}", use_container_width=True):
            logger.info(f"User confirmed deletion for project {pid} (Name: '{pname}') from Home page.")

            # Attempt deletion from database
            deleted_db = delete_project(pid)
            if deleted_db:
                st.success(f"Project '{pname}' deleted from database.")
                logger.info(f"Successfully deleted project {pid} from database.")
            else:
                st.error(f"Failed to delete project '{pname}' from database.")
                logger.error(f"Failed to delete project {pid} from database.")

            # Attempt to cleanup associated files and directories
            # cleanup_project_directories logs its own success/failure details
            cleanup_project_directories(pdir)

            st.session_state[CONFIRM_DELETE_STATE_KEY] = None  # Clear confirmation state
            st.rerun()  # Rerun to update the page (remove confirmation UI and potentially the project)

    with confirm_cols[1]:
        if st.button("❌ Cancel", key=f"confirm_del_no_{pid}", use_container_width=True):
            logger.info(f"User cancelled deletion for project {pid} (Name: '{pname}') from Home page.")
            st.session_state[CONFIRM_DELETE_STATE_KEY] = None  # Clear confirmation state
            st.rerun()  # Rerun to remove the confirmation UI


def _display_project_summary(project_header: dict):
    """Displays an expander summarizing a single project."""
    pid = project_header["id"]
    pname = project_header["name"]
    pdir = project_header["base_dir_path"]
    clip_count = project_header.get("clip_count", 0)
    created_at = project_header.get("created_at")

    # Format creation date safely
    created_str = "Unknown Date"
    if created_at:
        try:
            # Assuming created_at is an ISO format string
            created_dt = datetime.datetime.fromisoformat(created_at)
            created_str = created_dt.strftime("%Y-%m-%d %H:%M")
        except (ValueError, TypeError, AttributeError) as e:
            logger.warning(f"Could not parse created_at '{created_at}' for project {pid}: {e}")
            pass  # Keep "Unknown Date" if formatting fails

    # Get top clip info directly from the header (if available)
    top_clip_thumbnail = project_header.get("top_clip_thumbnail")
    top_clip_score = project_header.get("top_clip_score", 0)

    expander_title = f"**{pname}** ({clip_count} clips) - Created: {created_str}"

    # Use a unique key for the expander based on the project ID
    with st.expander(expander_title, key=f"home_expander_{pid}"):
        col_thumb, col_details, col_action = st.columns([1, 3, 1])

        # --- Thumbnail Column ---
        with col_thumb:
            if top_clip_thumbnail and os.path.exists(top_clip_thumbnail):
                try:
                    st.image(
                        top_clip_thumbnail,
                        use_column_width=True,
                        caption=f"Top Clip (Score: {top_clip_score:.2f})" if top_clip_score else "Top Clip",
                    )
                except Exception as img_err:
                    # Log error but don't crash the UI
                    logger.warning(f"Error loading thumbnail '{top_clip_thumbnail}' for project {pid}: {img_err}")
                    st.caption("⚠️ Thumbnail Error")
            else:
                st.caption("No Preview")

        # --- Details Column ---
        with col_details:
            source_type = project_header.get("source_type", "Unknown").capitalize()
            source_path = project_header.get("source_path", "N/A")

            st.markdown(f"**Source Type:** {source_type}")

            # Display source path appropriately
            if source_type == "Youtube":
                st.markdown(f"**URL:** [{source_path}]({source_path})")
            elif source_path != "N/A":
                # Show only filename for uploads, truncate long paths for display
                display_path = os.path.basename(source_path)
                if len(display_path) > 50:
                    display_path = "..." + display_path[-47:]
                st.markdown(f"**File:** {display_path}")
            else:
                st.markdown("**Source Path:** N/A")

            # Display highest score if clips exist
            if clip_count > 0:
                st.markdown(f"**Highest Score:** {top_clip_score:.2f}")
            else:
                st.markdown("_(No clips generated yet)_")

        # --- Action Column ---
        with col_action:
            # Button to navigate to the full project view
            if st.button("View Project", key=f"view_home_{pid}", use_container_width=True):
                st.session_state.current_project_id = pid
                st.session_state.app_mode = "📁 Projects"  # Switch app mode
                st.rerun()

            # Button to initiate deletion (triggers confirmation)
            if st.button("Delete Project", key=f"delete_home_action_{pid}", type="secondary", use_container_width=True):
                # Set state to trigger the confirmation dialog on the *next* rerun
                st.session_state[CONFIRM_DELETE_STATE_KEY] = {
                    "id": pid,
                    "name": pname,
                    "dir": pdir,
                }
                logger.debug(f"Initiating delete confirmation for project {pid} from home page.")
                st.rerun()


def _display_activity_summary(projects: list):
    """Calculates and displays overall activity metrics."""
    st.subheader("📊 Activity Summary")
    total_projects = len(projects)
    # Sum clip counts available in project headers
    total_clips_all_projects = sum(p.get("clip_count", 0) for p in projects)

    # Find the overall highest score from all loaded project headers
    all_top_scores = [p.get("top_clip_score", 0) for p in projects if p.get("top_clip_score") is not None]
    overall_highest_score = max(all_top_scores) if all_top_scores else 0

    # Display metrics in columns
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Projects", total_projects)
    col2.metric("Total Clips Generated", total_clips_all_projects)
    # Display score nicely formatted or N/A
    score_display = f"{overall_highest_score:.2f}" if overall_highest_score > 0 else "N/A"
    col3.metric("Highest Clip Score (Overall)", score_display)


def _display_how_it_works():
    """Displays the 'How It Works' and 'Features' sections."""
    st.info("👋 No projects yet. Create one to get started!", icon="💡")
    st.subheader("How It Works")
    st.markdown(
        """
    1.  **Upload** a video file or enter a YouTube link.
    2.  **Analyze** the content using AI to identify key moments, topics, and sentiment.
    3.  **Generate** short, engaging clips based on the analysis automatically.
    4.  **Review & Export** clips suitable for social media, presentations, or archives.

    *OpenClip Pro leverages AI to intelligently segment and evaluate your video content, saving you time in finding the best parts.*
    """
    )
    st.subheader("Features")
    features_cols = st.columns(2)
    with features_cols[0]:
        st.markdown(
            """
        - **AI-Powered Analysis**: Multiple AI models.
        - **Automatic Clip Generation**: Fast & efficient.
        - **Social Media Ready**: Optimized exports.
        """
        )
    with features_cols[1]:
        st.markdown(
            """
        - **Custom Scoring**: Identify engaging moments.
        - **Board of AI Directors**: Diverse perspectives.
        - **Detailed Analytics**: Understand content appeal.
        """
        )


def show():
    """Displays the Home dashboard page."""
    st.title("🏠 OpenClip Pro Dashboard")
    st.markdown("##### AI-Powered Video Analysis & Clipping")

    # --- Navigation Buttons ---
    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ New Project", use_container_width=True):
            st.session_state.app_mode = "🎬 Create New"
            st.rerun()
    with col2:
        if st.button("📁 View All Projects", use_container_width=True):
            st.session_state.app_mode = "📁 Projects"
            st.rerun()

    st.divider()

    # --- Deletion Confirmation Handling ---
    # Check if a delete confirmation was triggered from this page
    if CONFIRM_DELETE_STATE_KEY in st.session_state and st.session_state[CONFIRM_DELETE_STATE_KEY]:
        project_to_delete = st.session_state[CONFIRM_DELETE_STATE_KEY]
        _display_delete_confirmation(project_to_delete)
        # Stop rendering the rest of the page while confirmation is active
        # The rerun in _display_delete_confirmation handles moving away from this state.
        return

    # --- Load Project Data ---
    # Load project headers (summary info, not full details)
    # Assumes load_projects returns a list of dicts, sorted newest first,
    # and includes essential fields like id, name, base_dir_path, clip_count,
    # created_at, top_clip_thumbnail, top_clip_score.
    projects = load_projects() # Consider adding error handling for load_projects if needed

    if projects:
        # --- Display Metrics and Recent Projects ---
        _display_activity_summary(projects)
        st.subheader("📂 Recent Projects")
        # Display the specified number of most recent projects
        for project_header in projects[:MAX_RECENT_PROJECTS]:
            _display_project_summary(project_header)
    else:
        # --- Display Welcome/Instructions if no projects exist ---
        _display_how_it_works()

