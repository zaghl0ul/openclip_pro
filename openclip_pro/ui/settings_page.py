import streamlit as st
import logging
from database import get_settings, save_setting  # Assuming these functions interact with the DB

# Set up logger for this module
logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_PROJECT_NAME = "Project %Y-%m-%d"
DEFAULT_THEME = "dark"
DEFAULT_CLIP_LENGTH = 60
DEFAULT_FRAME_SAMPLE_RATE = 2.5
DEFAULT_SCORE_THRESHOLD = 75
DEFAULT_COMPRESSION_QUALITY = 85
DEFAULT_MAX_RESOLUTION = 720
DEFAULT_WORKERS_EXTRACTION = 4
DEFAULT_WORKERS_ENCODING = 4
DEFAULT_WORKERS_API = 8
DEFAULT_WORKERS_CLIP_GEN = 4
DEFAULT_EXPORT_FORMAT = "web_optimized"

THEME_OPTIONS = ["dark", "light"]
VALID_RESOLUTIONS = [256, 480, 512, 720, 1024, 1080]  # Allowed max resolutions for AI input
VALID_EXPORT_FORMATS = ["web_optimized", "high_quality", "copy_only"]

# --- Helper Functions ---

def get_setting_typed(settings, key, default, target_type=str, options=None):
    """
    Safely retrieves a setting, converts it to the target type,
    and ensures it's valid if options are provided.

    Args:
        settings (dict): The dictionary of current settings.
        key (str): The setting key to retrieve.
        default: The default value if the key is missing or invalid.
        target_type (type): The type to convert the value to (e.g., int, float, str).
        options (list, optional): A list of valid options for the setting.

    Returns:
        The retrieved, converted, and validated setting value.
    """
    value_str = settings.get(key)
    if value_str is None:
        return default

    try:
        typed_value = target_type(value_str)
        if options and typed_value not in options:
            logger.warning(
                f"Invalid value '{typed_value}' for setting '{key}'. "
                f"Falling back to default '{default}'."
            )
            # If numeric options, try finding the closest valid one
            if all(isinstance(opt, (int, float)) for opt in options):
                try:
                    return min(options, key=lambda x: abs(x - typed_value))
                except TypeError: # Handle cases where typed_value might not be comparable
                    return default
            else:
                return default # Fallback for non-numeric or mixed options
        return typed_value
    except (ValueError, TypeError) as e:
        logger.warning(
            f"Could not convert setting '{key}' value '{value_str}' to {target_type}. "
            f"Error: {e}. Falling back to default '{default}'."
        )
        return default

# --- Main Settings Page Function ---

def show():
    """Displays the main application settings page with various configuration tabs."""
    st.title("⚙️ Application Settings")
    st.caption("Configure default behaviors and analysis parameters for new projects.")

    # Retrieve necessary objects from session state
    ai_module = st.session_state.get("ai_module")
    model_registry = st.session_state.get("model_registry")

    # Fetch all current global settings from the database
    current_settings = get_settings()

    # --- Define Tabs ---
    tab_gen, tab_analysis, tab_export, tab_advanced = st.tabs(
        ["General Defaults", "Analysis Defaults", "Export Defaults", "Advanced"]
    )

    # --- General Defaults Tab ---
    with tab_gen:
        st.subheader("General Defaults")

        # Default Project Name Template
        default_project_name_value = get_setting_typed(
            current_settings, "default_project_name", DEFAULT_PROJECT_NAME, str
        )
        default_project_name = st.text_input(
            "Default Project Name Template",
            value=default_project_name_value,
            help="Template for new project names. Use date/time format codes like %Y, %m, %d, %H, %M.",
        )

        st.subheader("Appearance (Requires App Restart)")
        st.warning(
            "Note: Theme settings require restarting the Streamlit application to take full effect."
        )

        # Theme Setting
        current_theme_setting = get_setting_typed(
            current_settings, "default_theme", DEFAULT_THEME, str, THEME_OPTIONS
        )
        theme_index = THEME_OPTIONS.index(current_theme_setting)
        theme = st.selectbox(
            "Theme", THEME_OPTIONS, index=theme_index, key="theme_selectbox"
        )

        # Save Button for General Settings
        if st.button("Save General Settings", key="save_gen_settings"):
            saved_any = False
            # save_setting returns True if the value was changed in the DB
            if save_setting("default_project_name", default_project_name):
                saved_any = True
            if save_setting("default_theme", theme.lower()): # Save theme as lowercase
                saved_any = True

            if saved_any:
                st.success("General settings saved!")
                # Force a reload of settings into session state if needed immediately elsewhere
                st.session_state.user_settings = get_settings()
                st.rerun() # Rerun to reflect saved state and show success message clearly
            else:
                st.info("No changes detected in general settings.")

    # --- Analysis Defaults Tab ---
    with tab_analysis:
        st.subheader("Default Initial Analysis Parameters")
        st.caption("These values will pre-fill the 'Create New Project' form.")

        analysis_cols = st.columns(2)

        # --- Basic Analysis Parameters ---
        default_clip_length_value = get_setting_typed(
            current_settings, "default_clip_length", DEFAULT_CLIP_LENGTH, int
        )
        default_frame_sample_rate_value = get_setting_typed(
            current_settings, "default_frame_sample_rate", DEFAULT_FRAME_SAMPLE_RATE, float
        )
        default_score_threshold_value = get_setting_typed(
            current_settings, "default_score_threshold", DEFAULT_SCORE_THRESHOLD, int
        )

        with analysis_cols[0]:
            default_clip_length = st.slider(
                "Segment Length (s)", min_value=10, max_value=180,
                value=default_clip_length_value, step=5, key="set_def_clip_len",
                help="Default length of video segments processed during analysis."
            )
            default_frame_sample_rate = st.slider(
                "Frame Sample Rate (s)", min_value=0.5, max_value=10.0,
                value=default_frame_sample_rate_value, step=0.5, key="set_def_sample_rate",
                help="Default interval between frames sampled for AI analysis."
            )
        with analysis_cols[1]:
            default_score_threshold = st.slider(
                "Score Threshold", min_value=0, max_value=100,
                value=default_score_threshold_value, step=5, key="set_def_score",
                help="Default minimum AI confidence score to consider a finding relevant."
            )

        st.divider() # Visually separate sections

        # --- Default AI Model ---
        st.markdown("**Default AI Model for Analysis**")

        # Initialize variables to store selections or current settings
        selected_ai_provider = None
        selected_ai_model = None
        ai_selection_possible = bool(ai_module and model_registry)

        if ai_selection_possible:
            providers = model_registry.list_providers()
            if not providers:
                st.warning("No AI providers available. Check AI module configuration and dependencies.")
                # Keep previously saved settings if any, but disable selection
                selected_ai_provider = get_setting_typed(current_settings, "default_ai_provider", "N/A", str)
                selected_ai_model = get_setting_typed(current_settings, "default_ai_model", "N/A", str)
                ai_selection_possible = False # Prevent saving if no providers found now
            else:
                # Get current provider setting, fallback to the first available provider
                current_provider_setting = get_setting_typed(
                    current_settings, "default_ai_provider", providers[0], str, providers
                )
                prov_index = providers.index(current_provider_setting)

                selected_ai_provider = st.selectbox(
                    "Default AI Provider", providers, index=prov_index, key="set_def_ai_provider"
                )

                # List models for the *currently selected* provider in the UI
                models = model_registry.list_models_for_provider(selected_ai_provider)
                if not models:
                    st.warning(
                        f"No models found for '{selected_ai_provider}'. Check provider setup (e.g., API keys)."
                    )
                    # Keep previously saved model if any, but disable selection for this provider
                    selected_ai_model = get_setting_typed(current_settings, "default_ai_model", "N/A", str)
                    ai_selection_possible = False # Prevent saving invalid combo
                else:
                    # Get current model setting, fallback to first available model for the selected provider
                    current_model_setting = get_setting_typed(
                         current_settings, "default_ai_model", models[0], str, models
                    )
                    # If the saved model isn't valid for the *currently selected* provider, use the first valid model
                    if current_model_setting not in models:
                        current_model_setting = models[0]

                    mod_index = models.index(current_model_setting)
                    selected_ai_model = st.selectbox(
                        "Default AI Model", models, index=mod_index, key="set_def_ai_model"
                    )
        else:
            # AI Module or Registry not available
            st.warning(
                "AI Module or Model Registry not loaded. Cannot select default AI model."
            )
            # Display saved values if they exist, but disable interaction
            selected_ai_provider = get_setting_typed(current_settings, "default_ai_provider", "N/A", str)
            selected_ai_model = get_setting_typed(current_settings, "default_ai_model", "N/A", str)

        # Display provider/model (disabled if selection wasn't possible)
        st.text_input(
            "Selected AI Provider", value=selected_ai_provider or "N/A", disabled=True
        )
        st.text_input(
            "Selected AI Model", value=selected_ai_model or "N/A", disabled=True
        )

        st.divider()

        # --- Image Processing for AI ---
        st.subheader("Image Processing for AI Analysis")
        img_proc_cols = st.columns(2)

        compression_quality_value = get_setting_typed(
            current_settings, "compression_quality", DEFAULT_COMPRESSION_QUALITY, int
        )
        # Ensure value is within slider range (although get_setting_typed handles invalid options)
        compression_quality_value = max(50, min(100, compression_quality_value))

        max_resolution_value = get_setting_typed(
            current_settings, "max_resolution", DEFAULT_MAX_RESOLUTION, int, VALID_RESOLUTIONS
        )

        with img_proc_cols[0]:
            compression_quality = st.slider(
                "JPEG Quality (AI Input)", min_value=50, max_value=100,
                value=compression_quality_value, step=5, key="set_jpeg_qual",
                help="Quality setting (50-100) for encoding frames sent to AI models. Lower quality reduces size but may impact accuracy."
            )
        with img_proc_cols[1]:
            max_resolution = st.select_slider(
                "Max Resolution (AI Input)", options=VALID_RESOLUTIONS,
                value=max_resolution_value, key="set_max_res",
                help="Maximum dimension (width or height) for frames sent to AI. Larger images may improve detail but increase cost/latency and risk hitting token limits."
            )

        st.divider()

        # --- Concurrency Settings ---
        st.subheader("Concurrency Settings")
        st.caption(
            "Adjust worker counts for parallel tasks. Higher values may speed up processing but increase CPU/memory/network usage."
        )
        concurrency_cols = st.columns(4)

        max_workers_extraction_value = get_setting_typed(
            current_settings, "max_workers_extraction", DEFAULT_WORKERS_EXTRACTION, int
        )
        max_workers_encoding_value = get_setting_typed(
            current_settings, "max_workers_encoding", DEFAULT_WORKERS_ENCODING, int
        )
        max_workers_api_value = get_setting_typed(
            current_settings, "max_workers_api", DEFAULT_WORKERS_API, int
        )
        max_workers_clip_gen_value = get_setting_typed(
            current_settings, "max_workers_clip_gen", DEFAULT_WORKERS_CLIP_GEN, int
        )

        with concurrency_cols[0]:
            max_workers_extraction = st.number_input(
                "Frame Extract Workers", min_value=1, max_value=16,
                value=max_workers_extraction_value, step=1, key="set_conc_extract",
                help="Parallel workers for reading frames from video files (CPU/Disk bound)."
            )
        with concurrency_cols[1]:
            max_workers_encoding = st.number_input(
                "Frame Encode Workers", min_value=1, max_value=16,
                value=max_workers_encoding_value, step=1, key="set_conc_encode",
                help="Parallel workers for encoding extracted frames to JPEG/base64 (CPU bound)."
            )
        with concurrency_cols[2]:
            max_workers_api = st.number_input(
                "Analysis API Workers", min_value=1, max_value=16, # Max might depend on API limits
                value=max_workers_api_value, step=1, key="set_conc_api",
                help="Parallel workers for sending requests to the AI API (Network/API bound)."
            )
        with concurrency_cols[3]:
            max_workers_clip_gen = st.number_input(
                "Clip Gen Workers", min_value=1, max_value=16,
                value=max_workers_clip_gen_value, step=1, key="set_conc_clipgen",
                help="Parallel workers for running ffmpeg to generate clip video files (CPU bound)."
            )

        # Save Button for Analysis Settings
        if st.button("Save Analysis Defaults", key="save_analysis_settings"):
            saved_any = False
            # Note: Saving numeric settings as strings, assuming save_setting handles/expects this.
            if save_setting("default_clip_length", str(default_clip_length)): saved_any = True
            if save_setting("default_frame_sample_rate", str(default_frame_sample_rate)): saved_any = True
            if save_setting("default_score_threshold", str(default_score_threshold)): saved_any = True

            # Only save AI model settings if selection was possible and valid values were selected
            if ai_selection_possible and selected_ai_provider and selected_ai_model and selected_ai_provider != "N/A" and selected_ai_model != "N/A":
                if save_setting("default_ai_provider", selected_ai_provider): saved_any = True
                if save_setting("default_ai_model", selected_ai_model): saved_any = True
            elif not ai_selection_possible and ai_module:
                # Log if we couldn't save because options were unavailable
                logger.warning(
                    "Could not save default AI Provider/Model settings because "
                    "valid options were unavailable or AI module failed."
                )

            # Save image processing and concurrency settings
            if save_setting("compression_quality", str(compression_quality)): saved_any = True
            if save_setting("max_resolution", str(max_resolution)): saved_any = True
            if save_setting("max_workers_extraction", str(max_workers_extraction)): saved_any = True
            if save_setting("max_workers_encoding", str(max_workers_encoding)): saved_any = True
            if save_setting("max_workers_api", str(max_workers_api)): saved_any = True
            if save_setting("max_workers_clip_gen", str(max_workers_clip_gen)): saved_any = True

            if saved_any:
                st.success("Default analysis settings saved!")
                st.session_state.user_settings = get_settings() # Reload settings into state
                st.rerun()
            else:
                st.info("No changes detected in analysis settings.")

    # --- Export Defaults Tab ---
    with tab_export:
        st.subheader("Default Export Settings")
        st.caption("These settings determine the default options when exporting clips.")

        current_format = get_setting_typed(
            current_settings, "export_format", DEFAULT_EXPORT_FORMAT, str, VALID_EXPORT_FORMATS
        )
        format_index = VALID_EXPORT_FORMATS.index(current_format)

        export_format = st.selectbox(
            "Default Clip Encoding Format",
            VALID_EXPORT_FORMATS,
            index=format_index,
            key="set_def_export_fmt",
            help=(
                "'web_optimized': Good balance of size/quality (e.g., H.264/AAC in MP4). "
                "'high_quality': Less compression, larger files (e.g., ProRes or high-bitrate H.264). "
                "'copy_only': Fastest, uses original video/audio streams without re-encoding (may have compatibility issues)."
            )
        )
        st.caption(
            "Note: Platform-specific export optimizations (e.g., for social media) "
            "are typically available on the Project Export page itself."
        )

        # Save Button for Export Settings
        if st.button("Save Export Defaults", key="save_export_settings"):
            if save_setting("export_format", export_format):
                st.success("Default export settings saved!")
                st.session_state.user_settings = get_settings() # Reload settings into state
                st.rerun()
            else:
                st.info("No changes detected in export settings.")

    # --- Advanced Tab ---
    with tab_advanced:
        st.subheader("Advanced Settings")
        st.warning("🚧 No advanced settings are currently available.")
        # Placeholder for future settings like:
        # - Explicit ffmpeg path override
        # - Application logging level configuration
        # - Toggling experimental features (feature flags)
        # - Cache clearing options (though often requires app restart)

# --- Entry Point (if run directly, though typically imported) ---
if __name__ == "__main__":
    # Example of how to run this if it were the main app page
    # This requires setting up dummy session state and DB functions for testing
    st.set_page_config(layout="wide")

    # Mock database functions if needed for standalone testing
    if 'user_settings' not in st.session_state:
        st.session_state.user_settings = {} # Initialize dummy settings

    def mock_get_settings():
        # Return defaults or load from a dummy dict
        return st.session_state.user_settings

    def mock_save_setting(key, value):
        # Simulate saving and check if value changed
        old_value = st.session_state.user_settings.get(key)
        if old_value != value:
            st.session_state.user_settings[key] = value
            logger.info(f"Mock DB: Saved {key} = {value}")
            return True
        return False

    # Replace real DB functions with mocks for this example run
    get_settings = mock_get_settings
    save_setting = mock_save_setting

    # Mock AI module and registry if needed for testing AI model selection
    if 'ai_module' not in st.session_state:
        class MockModelRegistry:
            def list_providers(self): return ["mock_provider_A", "mock_provider_B"]
            def list_models_for_provider(self, provider):
                if provider == "mock_provider_A": return ["model_A1", "model_A2"]
                if provider == "mock_provider_B": return ["model_B1"]
                return []
        st.session_state.ai_module = True # Just needs to exist
        st.session_state.model_registry = MockModelRegistry()

    show()
