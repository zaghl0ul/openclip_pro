import streamlit as st
import datetime
import os
import time
import logging

from database import load_projects, delete_project
from media_utils import cleanup_project_directories

logger = logging.getLogger(__name__)

# --- Constants ---
SESSION_STATE_PROJECT_SEARCH = "project_search"
SESSION_STATE_PROJECT_SORT = "projects_list_sort_by"
SESSION_STATE_PROJECT_VIEW = "projects_list_view_mode"
SESSION_STATE_CONFIRM_DELETE = "confirm_delete_project_info"
SESSION_STATE_CURRENT_PROJECT_ID = "current_project_id"
SESSION_STATE_APP_MODE = "app_mode"

DEFAULT_SORT_OPTION = "Newest First"
DEFAULT_VIEW_MODE = "List"

SORT_OPTIONS = [
    "Newest First",
    "Oldest First",
    "Name (A-Z)",
    "Clip Count",
    "Top Score (High-Low)",
    "Top Score (Low-High)",
]
VIEW_OPTIONS = ["List", "Gallery"]


# --- Helper Functions ---

def _initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if SESSION_STATE_PROJECT_SORT not in st.session_state:
        st.session_state[SESSION_STATE_PROJECT_SORT] = DEFAULT_SORT_OPTION
    if SESSION_STATE_PROJECT_VIEW not in st.session_state:
        st.session_state[SESSION_STATE_PROJECT_VIEW] = DEFAULT_VIEW_MODE
    if SESSION_STATE_PROJECT_SEARCH not in st.session_state:
        st.session_state[SESSION_STATE_PROJECT_SEARCH] = ""
    if SESSION_STATE_CONFIRM_DELETE not in st.session_state:
        st.session_state[SESSION_STATE_CONFIRM_DELETE] = None


def _filter_and_sort_projects(projects, search_term, sort_by):
    """Filter and sort the list of projects based on user input."""
    # Apply filtering
    filtered_projects = projects
    if search_term:
        search_term_lower = search_term.lower()
        filtered_projects = [
            p for p in filtered_projects
            if search_term_lower in p.get("name", "").lower()
        ]

    # Determine sorting key and direction
    sort_key_func = lambda x: x.get("created_at", "")  # Default: Newest First
    reverse_sort = True

    if sort_by == "Oldest First":
        reverse_sort = False
    elif sort_by == "Name (A-Z)":
        sort_key_func = lambda x: x.get("name", "").lower()
        reverse_sort = False
    elif sort_by == "Clip Count":
        # Sort by clip count descending, handle potential None with default 0
        sort_key_func = lambda x: x.get("clip_count", 0) or 0
        reverse_sort = True
    elif sort_by == "Top Score (High-Low)":
        # Sort by score descending, handle potential None with default 0
        sort_key_func = lambda x: x.get("top_clip_score", 0) or 0
        reverse_sort = True
    elif sort_by == "Top Score (Low-High)":
        # Sort by score ascending, handle potential None with default 0
        sort_key_func = lambda x: x.get("top_clip_score", 0) or 0
        reverse_sort = False

    # Apply sorting with error handling
    try:
        # Ensure sorting handles None values gracefully by using the default in lambda
        sorted_projects = sorted(filtered_projects, key=sort_key_func, reverse=reverse_sort)
    except Exception as e:
        logger.error(f"Error during project list sorting: {e}", exc_info=True)
        st.warning(f"Could not apply sort order '{sort_by}'. Displaying default order.")
        sorted_projects = filtered_projects # Fallback to unsorted list

    return sorted_projects

def _format_datetime(iso_string):
    """Safely format an ISO datetime string."""
    if not iso_string:
        return "Unknown"
    try:
        return datetime.datetime.fromisoformat(iso_string).strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        logger.warning(f"Could not parse date: {iso_string}", exc_info=False)
        return "Invalid Date"

def _display_project_thumbnail(thumbnail_path, width=100, use_column_width=False):
    """Displays a project thumbnail image if available."""
    if thumbnail_path and os.path.exists(thumbnail_path):
        try:
            st.image(thumbnail_path, width=width if not use_column_width else None, use_column_width=use_column_width)
        except Exception as e:
            logger.error(f"Error displaying image {thumbnail_path}: {e}", exc_info=True)
            st.caption("Preview Err") # Show error if image fails to load
    else:
        st.caption("No Preview") # Placeholder if no thumbnail

def _handle_delete_action(project_id, project_name, project_dir):
    """Performs the actual deletion of a project and its files."""
    logger.info(f"Attempting deletion for project '{project_name}' (ID: {project_id})")
    deleted_db = delete_project(project_id) # Delete from DB
    cleanup_project_directories(project_dir) # Cleanup files (logs internally)

    if deleted_db:
        st.success(f"Project '{project_name}' deleted from database.")
    else:
        # delete_project should ideally raise an exception on failure,
        # but we handle the boolean return case as per original code.
        st.error(f"Failed to delete project '{project_name}' from database.")

    # Clear confirmation flag after action
    st.session_state[SESSION_STATE_CONFIRM_DELETE] = None
    # Add a small delay before rerun for user feedback
    time.sleep(1.5)
    st.rerun()

def _display_deletion_confirmation():
    """Displays the confirmation dialog for project deletion."""
    project_info = st.session_state.get(SESSION_STATE_CONFIRM_DELETE)
    if not project_info:
        return False # No confirmation active

    pid = project_info["id"]
    pname = project_info["name"]
    pdir = project_info["dir"]
    view_mode_origin = project_info["view"] # Track where delete was clicked

    st.warning(f"⚠️ Confirm permanent deletion of project '{pname}'?")
    st.caption("This action is irreversible and will delete the database entry and associated files/folders.")

    confirm_cols = st.columns(2)
    with confirm_cols[0]:
        # Unique key incorporating view mode and ID for robustness
        if st.button("✅ Yes, Delete", key=f"confirm_del_{view_mode_origin}_{pid}", type="primary"):
            _handle_delete_action(pid, pname, pdir)
            # Rerun happens within _handle_delete_action

    with confirm_cols[1]:
        if st.button("❌ Cancel", key=f"cancel_del_{view_mode_origin}_{pid}"):
            logger.info(f"User cancelled deletion for project {pid} from {view_mode_origin} view.")
            st.session_state[SESSION_STATE_CONFIRM_DELETE] = None
            st.rerun()

    return True # Confirmation is being displayed

def _display_list_view(projects):
    """Renders the projects in a list format."""
    st.subheader("Project List")
    st.markdown("---")

    # Check if deletion confirmation is active for *this* view
    # We display it *before* the list for prominence
    confirmation_active = _display_deletion_confirmation()
    if confirmation_active and st.session_state[SESSION_STATE_CONFIRM_DELETE]["view"] == "list":
         return # Don't render the list while confirming for list view

    for project in projects:
        pid = project["id"]
        pname = project.get("name", "Unnamed Project")
        pdir = project["base_dir_path"] # Needed for cleanup
        clip_count = project.get("clip_count", 0)
        source_type = project.get("source_type", "N/A").capitalize()
        created_str = _format_datetime(project.get("created_at"))
        top_clip_thumbnail = project.get("top_clip_thumbnail")
        top_clip_score = project.get("top_clip_score", 0)

        item_cols = st.columns([1, 3, 1, 1]) # Thumbnail, Info, Open, Delete

        with item_cols[0]:
            _display_project_thumbnail(top_clip_thumbnail, width=100)

        with item_cols[1]:
            st.markdown(f"**{pname}**")
            st.caption(f"Created: {created_str} | Source: {source_type}")
            # Display score only if clips exist
            score_display = f"{top_clip_score:.2f}" if clip_count > 0 and top_clip_score is not None else "N/A"
            st.caption(f"Clips: {clip_count} | Top Score: {score_display}")

        with item_cols[2]:
            if st.button("Open", key=f"open_list_{pid}", use_container_width=True):
                st.session_state[SESSION_STATE_CURRENT_PROJECT_ID] = pid
                # Main app router should handle navigation based on current_project_id
                st.rerun()
        with item_cols[3]:
            if st.button("Delete", key=f"delete_list_{pid}", type="secondary", use_container_width=True):
                # Set confirmation details, including the view it originated from
                st.session_state[SESSION_STATE_CONFIRM_DELETE] = {
                    "id": pid, "name": pname, "dir": pdir, "view": "list"
                }
                st.rerun()

        st.markdown("---") # Separator for list items


def _display_gallery_view(projects, columns=4):
    """Renders the projects in a gallery format."""
    st.subheader("Project Gallery")
    st.markdown("---")

    # Check if deletion confirmation is active for *this* view
    # We display it *before* the gallery for prominence
    confirmation_active = _display_deletion_confirmation()
    if confirmation_active and st.session_state[SESSION_STATE_CONFIRM_DELETE]["view"] == "gallery":
         return # Don't render the gallery while confirming for gallery view

    cols = st.columns(columns)
    col_index = 0

    for project in projects:
        pid = project["id"]
        pname = project.get("name", "Unnamed Project")
        pdir = project["base_dir_path"]
        clip_count = project.get("clip_count", 0)
        # Use shorter date format for gallery
        created_at = project.get("created_at")
        created_str = "Unknown"
        if created_at:
             try:
                 created_str = datetime.datetime.fromisoformat(created_at).strftime("%Y-%m-%d")
             except (ValueError, TypeError):
                 pass # Keep "Unknown"

        top_clip_thumbnail = project.get("top_clip_thumbnail")
        top_clip_score = project.get("top_clip_score", 0)

        # Place the project card in the current column
        with cols[col_index % columns]:
            # Use a container with a border for card-like appearance
            with st.container(border=True):
                st.markdown(f"**{pname}**")
                _display_project_thumbnail(top_clip_thumbnail, use_column_width=True)

                score_display = f"{top_clip_score:.2f}" if clip_count > 0 and top_clip_score is not None else "N/A"
                st.caption(f"Clips: {clip_count} | Top: {score_display} | {created_str}")

                # Buttons below the info
                btn_cols = st.columns(2)
                with btn_cols[0]:
                    if st.button("Open", key=f"open_gal_{pid}", use_container_width=True):
                        st.session_state[SESSION_STATE_CURRENT_PROJECT_ID] = pid
                        st.rerun()
                with btn_cols[1]:
                    if st.button("Del", key=f"delete_gal_{pid}", type="secondary", use_container_width=True, help="Delete Project"):
                        # Set confirmation details, including the view it originated from
                        st.session_state[SESSION_STATE_CONFIRM_DELETE] = {
                            "id": pid, "name": pname, "dir": pdir, "view": "gallery"
                        }
                        st.rerun()
        col_index += 1 # Move to the next column for the next project


# --- Main Page Function ---

def show():
    """Displays the Projects page, allowing users to view, sort, filter, and manage projects."""
    st.title("📁 Projects")

    # Ensure session state variables are initialized
    _initialize_session_state()

    # Load project headers (includes basic info, counts, top clip thumbnail/score)
    projects = load_projects()

    # Handle case where no projects exist
    if not projects:
        st.info("No projects found. Create a new project to get started!", icon="💡")
        if st.button("➕ Create New Project"):
            st.session_state[SESSION_STATE_APP_MODE] = "🎬 Create New"
            st.rerun()
        return

    # --- Filtering, Sorting, and View Controls ---
    col_search, col_sort, col_view = st.columns([2, 1, 1])

    with col_search:
        # Use session state for persistent search term
        search_term = st.text_input(
            "Search by Name",
            st.session_state[SESSION_STATE_PROJECT_SEARCH],
            key="project_search_input", # Use a different key for the widget itself
            placeholder="Enter project name...",
            on_change=lambda: st.session_state.update({SESSION_STATE_PROJECT_SEARCH: st.session_state.project_search_input})
        )
        # Update session state immediately if using on_change is not preferred or causing issues
        # st.session_state[SESSION_STATE_PROJECT_SEARCH] = search_term


    with col_sort:
        # Use session state for persistent sort option
        current_sort = st.session_state[SESSION_STATE_PROJECT_SORT]
        sort_by = st.selectbox(
            "Sort by",
            SORT_OPTIONS,
            index=SORT_OPTIONS.index(current_sort), # Find index of current sort option
            key="project_sort_select",
        )
        # Update session state if the sort selection changes
        if sort_by != current_sort:
            st.session_state[SESSION_STATE_PROJECT_SORT] = sort_by
            st.rerun() # Rerun to apply new sort order

    with col_view:
        # Use session state for persistent view mode
        current_view = st.session_state[SESSION_STATE_PROJECT_VIEW]
        view_index = VIEW_OPTIONS.index(current_view)
        view_mode = st.radio(
            "View As",
            VIEW_OPTIONS,
            index=view_index,
            horizontal=True,
            key="project_view_mode_radio",
        )
        # Update session state and rerun if the view mode changes
        if view_mode != current_view:
            st.session_state[SESSION_STATE_PROJECT_VIEW] = view_mode
            st.rerun()

    # --- Filter and Sort Projects ---
    # Pass the current search term from session state
    filtered_sorted_projects = _filter_and_sort_projects(
        projects,
        st.session_state[SESSION_STATE_PROJECT_SEARCH],
        st.session_state[SESSION_STATE_PROJECT_SORT]
    )

    # --- Display Header Info ---
    st.caption(f"Showing {len(filtered_sorted_projects)} of {len(projects)} projects.")
    st.markdown("---") # Separator before project display

    # --- Display Projects based on View Mode ---
    if not filtered_sorted_projects:
        st.info("No projects match your current filters or search criteria.")
    else:
        view_mode = st.session_state[SESSION_STATE_PROJECT_VIEW]
        if view_mode == "List":
            _display_list_view(filtered_sorted_projects)
        elif view_mode == "Gallery":
            _display_gallery_view(filtered_sorted_projects)
        else:
            # Fallback or error for unknown view mode
            st.error(f"Unknown view mode: {view_mode}")
            _display_list_view(filtered_sorted_projects) # Default to list view

    # --- Footer Button ---
    st.divider()
    if st.button("➕ Create New Project ", key="create_new_btn_bottom"):
        st.session_state[SESSION_STATE_APP_MODE] = "🎬 Create New"
        st.rerun()
