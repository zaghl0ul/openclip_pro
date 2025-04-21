# ai_ui.py
"""
Contains Streamlit UI generation functions related to AI provider/model
configuration and API key management.
"""

import logging
import streamlit as st
from typing import Dict, List, Optional, Any

# Import specific classes needed
from ai.ai_models import (
    ModelRegistry,
    APIKeyManager,
    fernet,  # Used for API key encryption status check
)

# Setup logging
logger = logging.getLogger(__name__)

# Constants for session state keys to prevent typos and manage state centrally
SESSION_KEY_AI_UI_PREFIX = "ai_ui_state_"
SESSION_KEY_ACTIVE_PROVIDER = SESSION_KEY_AI_UI_PREFIX + "active_provider"
SESSION_KEY_ACTIVE_MODEL = SESSION_KEY_AI_UI_PREFIX + "active_model"
SESSION_KEY_BOARD_ENABLED = SESSION_KEY_AI_UI_PREFIX + "board_enabled"
SESSION_KEY_BOARD_MEMBERS = SESSION_KEY_AI_UI_PREFIX + "board_members"
SESSION_KEY_BOARD_TASKS = SESSION_KEY_AI_UI_PREFIX + "board_tasks"
SESSION_KEY_CHAIRPERSON = SESSION_KEY_AI_UI_PREFIX + "chairperson"

# Define available task types for the AI Board
AVAILABLE_BOARD_TASKS = [
    "viral_analysis",
    "monetization_analysis",
    # Future tasks: "thumbnail_selection", "hook_analysis", "audience_analysis"
]

# --- Helper Functions ---

def _initialize_session_state_if_missing(key: str, default_value: Any):
    """Initializes a session state key if it doesn't exist."""
    if key not in st.session_state:
        st.session_state[key] = default_value

def _get_validated_selection(
    session_key: str, available_options: List[str], default_index: int = 0
) -> str:
    """
    Gets the current selection from session state, ensuring it's still valid.
    Updates session state if the stored value is invalid.
    """
    if not available_options:
        # If there are no options, return a default or handle appropriately
        # In this context (model/provider selection), caller should handle empty lists
        return ""

    current_value = st.session_state.get(session_key)

    if current_value not in available_options:
        # If stored value is invalid (e.g., provider/model removed), reset to default
        current_value = available_options[default_index]
        st.session_state[session_key] = current_value # Update session state
        logger.debug(f"Resetting session state key '{session_key}' to default '{current_value}' as previous value was invalid.")

    return current_value

def _format_model_id(provider: str, model: str) -> str:
    """Formats provider and model name into a single string identifier."""
    return f"{provider}:{model}"

def _parse_model_id(model_id: str) -> Optional[Dict[str, str]]:
    """Parses a model identifier string back into provider and model name."""
    try:
        provider_name, model_name = model_id.split(":", 1)
        return {"provider": provider_name, "model": model_name}
    except ValueError:
        logger.warning(f"Invalid model format skipped: {model_id}")
        return None

# --- Main UI Functions ---

def create_model_selection_ui(
    model_registry: ModelRegistry, key_manager: APIKeyManager
) -> Dict[str, Any]:
    """
    Renders UI for selecting AI models and board configuration in Streamlit sidebar.

    Uses st.session_state to persist selections across reruns.

    Args:
        model_registry: An instance of ModelRegistry containing available models.
        key_manager: An instance of APIKeyManager for checking API key status.

    Returns:
        Dictionary containing the selected primary model configuration and
        board configuration, reflecting the current state. Note that
        st.session_state is the primary source of truth for the UI state.
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### AI Model Configuration")

    # --- Primary Model Selection ---
    st.sidebar.subheader("Primary Analysis Model")
    available_providers = model_registry.list_providers()

    if not available_providers:
        st.sidebar.warning(
            "No AI providers available. Check dependencies and console logs."
        )
        # Return a config indicating no selection possible
        return {
            "primary": {"provider": None, "model": None},
            "board_enabled": False,
            "board_members": [],
            "chairperson": None,
            "board_tasks": [],
        }

    # Initialize session state for provider if needed (defaults to first provider)
    _initialize_session_state_if_missing(SESSION_KEY_ACTIVE_PROVIDER, available_providers[0])

    # Get validated provider selection from session state
    selected_provider = _get_validated_selection(SESSION_KEY_ACTIVE_PROVIDER, available_providers)
    provider_index = available_providers.index(selected_provider)

    # Provider Selectbox - updates session state via its key/onChange
    new_selected_provider = st.sidebar.selectbox(
        "Select AI Provider",
        available_providers,
        index=provider_index,
        key="primary_provider_select", # Use key for implicit state management is fine
        # Using on_change explicitly updates our primary state variable if needed,
        # but Streamlit's key mechanism often handles this implicitly for selectbox.
        # For complex interactions, on_change can be clearer.
        # on_change=lambda: st.session_state.update({SESSION_KEY_ACTIVE_PROVIDER: st.session_state.primary_provider_select})
    )
    # Ensure our main session state variable reflects the selectbox's current value
    st.session_state[SESSION_KEY_ACTIVE_PROVIDER] = new_selected_provider
    selected_provider = new_selected_provider # Use the potentially updated value

    # Get models for the *currently selected* provider
    available_models = model_registry.list_models_for_provider(selected_provider)

    primary_config = {"provider": selected_provider, "model": None} # Default if no models

    if not available_models:
        st.sidebar.warning(
            f"No models listed for provider '{selected_provider}'. Check ModelRegistry."
        )
    else:
        # Initialize session state for model if needed (defaults to first model)
        _initialize_session_state_if_missing(SESSION_KEY_ACTIVE_MODEL, available_models[0])

        # Get validated model selection (relative to the selected provider)
        # Need to ensure the stored model is valid *for the current provider*
        current_model_selection = st.session_state.get(SESSION_KEY_ACTIVE_MODEL)
        if current_model_selection not in available_models:
             # If the stored model isn't valid for this provider, default to the first one
            selected_model = available_models[0]
            st.session_state[SESSION_KEY_ACTIVE_MODEL] = selected_model
            logger.debug(f"Resetting model selection for provider '{selected_provider}' to '{selected_model}'.")
        else:
            selected_model = current_model_selection

        model_index = available_models.index(selected_model)

        # Model Selectbox
        new_selected_model = st.sidebar.selectbox(
            "Select Model",
            available_models,
            index=model_index,
            key="primary_model_select",
        )
        # Update session state
        st.session_state[SESSION_KEY_ACTIVE_MODEL] = new_selected_model
        selected_model = new_selected_model # Use updated value

        primary_config["model"] = selected_model

        # Display model info and API key status
        model_info = model_registry.get_model_info(selected_provider, selected_model)
        if model_info:
            cap_str = ", ".join(model_info.get("capabilities", [])) or "N/A"
            st.sidebar.caption(
                f"Type: {model_info.get('type', 'N/A')}, "
                f"Quality: {model_info.get('quality', 'N/A')}"
            )
            st.sidebar.caption(f"Capabilities: {cap_str}")

            # Check API key requirement and availability
            if model_info.get("requires_api_key", True):
                if not key_manager.get_key(selected_provider):
                    st.sidebar.error(
                        f"API key needed for {selected_provider.capitalize()}! "
                        "Add/check key in 'API Keys' section."
                    )
                else:
                    st.sidebar.success(f"API key found for {selected_provider.capitalize()}.")
            else:
                st.sidebar.info(
                    f"API key not required for {selected_provider.capitalize()} model "
                    f"'{selected_model}' (e.g., local Ollama)."
                )

    # --- Board of Directors Configuration ---
    st.sidebar.subheader("AI Board of Directors")

    # Initialize board-related session state keys if they don't exist
    _initialize_session_state_if_missing(SESSION_KEY_BOARD_ENABLED, False)
    _initialize_session_state_if_missing(SESSION_KEY_BOARD_MEMBERS, [])
    _initialize_session_state_if_missing(SESSION_KEY_BOARD_TASKS, ["viral_analysis", "monetization_analysis"])
    _initialize_session_state_if_missing(SESSION_KEY_CHAIRPERSON, None)

    # Board Enable Toggle
    enable_board = st.sidebar.toggle(
        "Enable AI Board",
        value=st.session_state[SESSION_KEY_BOARD_ENABLED],
        key="enable_board_toggle",
        # Update session state directly when toggle changes
        on_change=lambda: st.session_state.update({SESSION_KEY_BOARD_ENABLED: st.session_state.enable_board_toggle})
    )
    # Ensure state variable reflects current toggle value
    st.session_state[SESSION_KEY_BOARD_ENABLED] = enable_board

    board_config = {"board_enabled": enable_board} # Start building board config dict

    if enable_board:
        st.sidebar.markdown("Select board members:")

        # Determine available models for the board (vision capable, key available if needed)
        providers_with_keys_lower = {p.lower() for p in key_manager.list_providers_with_keys()}
        vision_models_info = model_registry.list_vision_models() # List of dicts
        available_board_models_dicts = []

        for model_info in vision_models_info:
            provider = model_info["provider"]
            model_name = model_info["model"]
            requires_key = model_info.get("requires_api_key", True)

            # Include if it doesn't require a key OR if it does and the key exists
            if not requires_key or provider.lower() in providers_with_keys_lower:
                available_board_models_dicts.append({"provider": provider, "model": model_name})
            else:
                logger.debug(f"Skipping board candidate {provider}:{model_name} - API key required but not found.")

        if not available_board_models_dicts:
            st.sidebar.warning(
                "No suitable vision models available for the board. "
                "Check API keys (for cloud models) or Ollama connection (for local)."
            )
            # Reset board members and chairperson in state if no models are available
            st.session_state[SESSION_KEY_BOARD_MEMBERS] = []
            st.session_state[SESSION_KEY_CHAIRPERSON] = None
        else:
            # Prepare options and defaults for the multiselect widget
            available_board_options_str = sorted([
                _format_model_id(m["provider"], m["model"]) for m in available_board_models_dicts
            ])

            # Default selection comes from session state, ensure they are still valid options
            current_board_members_dicts = st.session_state[SESSION_KEY_BOARD_MEMBERS]
            valid_defaults_str = [
                _format_model_id(m["provider"], m["model"])
                for m in current_board_members_dicts
                if _format_model_id(m["provider"], m["model"]) in available_board_options_str
            ]

            # Board Member Multiselect
            selected_member_strings = st.sidebar.multiselect(
                "Board Members",
                options=available_board_options_str,
                default=valid_defaults_str,
                key="board_members_multiselect",
                # Use on_change to parse selections and update structured state
                on_change=lambda: st.session_state.update({
                    SESSION_KEY_BOARD_MEMBERS: [
                        m for m_str in st.session_state.board_members_multiselect
                        if (m := _parse_model_id(m_str)) is not None
                    ]
                })
            )
            # Ensure state reflects current selections after potential on_change update
            selected_board_members_dicts = st.session_state[SESSION_KEY_BOARD_MEMBERS]


            # --- Advanced Board Settings Expander ---
            with st.sidebar.expander("Advanced Board Settings"):

                # Chairperson Selection - Options are the currently selected members
                chairperson_options_str = sorted([
                    _format_model_id(m["provider"], m["model"]) for m in selected_board_members_dicts
                ])

                if chairperson_options_str:
                    # Default selection comes from session state, ensure it's a valid *selected* member
                    current_chairperson_dict = st.session_state.get(SESSION_KEY_CHAIRPERSON)
                    default_chair_str = None
                    if current_chairperson_dict:
                        chair_id = _format_model_id(current_chairperson_dict["provider"], current_chairperson_dict["model"])
                        if chair_id in chairperson_options_str:
                            default_chair_str = chair_id

                    # If no valid default, use the first option (index 0)
                    chair_index = chairperson_options_str.index(default_chair_str) if default_chair_str else 0

                    # Chairperson Selectbox
                    chairperson_selection_str = st.selectbox(
                        "Chairperson (Synthesizes Results)",
                        options=chairperson_options_str,
                        index=chair_index,
                        key="chairperson_select",
                         # Use on_change to parse selection and update structured state
                        on_change=lambda: st.session_state.update({
                            SESSION_KEY_CHAIRPERSON: _parse_model_id(st.session_state.chairperson_select)
                        })
                    )
                    # Ensure state reflects selection
                    st.session_state[SESSION_KEY_CHAIRPERSON] = _parse_model_id(chairperson_selection_str)

                else:
                    st.caption("Select board members above to choose a chairperson.")
                    # Ensure chairperson is cleared if no members are selected
                    st.session_state[SESSION_KEY_CHAIRPERSON] = None


                # Task Selection
                # Default selection comes from session state, ensure they are still valid task types
                current_tasks = st.session_state.get(SESSION_KEY_BOARD_TASKS, [])
                valid_default_tasks = [t for t in current_tasks if t in AVAILABLE_BOARD_TASKS]

                # Board Tasks Multiselect
                selected_tasks = st.multiselect(
                    "Board Tasks",
                    options=AVAILABLE_BOARD_TASKS,
                    default=valid_default_tasks,
                    key="board_tasks_multiselect",
                    # Update state on change
                    on_change=lambda: st.session_state.update({SESSION_KEY_BOARD_TASKS: st.session_state.board_tasks_multiselect})
                )
                # Ensure state reflects selection
                st.session_state[SESSION_KEY_BOARD_TASKS] = selected_tasks


    # --- Quick Setup Button ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Quick Setup")
    st.sidebar.caption("Quickly configure settings using only free models.")

    if st.sidebar.button("Use Only Free Models", key="use_free_models_button"):
        free_models_list = model_registry.list_free_models() # List of dicts

        if free_models_list:
            # Try to find a free vision model first for primary analysis
            free_vision_models = [
                m for m in free_models_list if "vision" in m.get("capabilities", []) or "image" in m.get("capabilities", [])
            ]

            primary_model_info = None
            if free_vision_models:
                # Sort by quality (e.g., premium > standard > fast) if available, else use first
                free_vision_models.sort(key=lambda x: {"premium": 0, "standard": 1, "fast": 2}.get(x.get("quality", "standard"), 1))
                primary_model_info = free_vision_models[0]
            elif free_models_list:
                 # Fallback to the first available free model if no vision model
                free_models_list.sort(key=lambda x: {"premium": 0, "standard": 1, "fast": 2}.get(x.get("quality", "standard"), 1))
                primary_model_info = free_models_list[0]

            if primary_model_info:
                # Set primary model state
                st.session_state[SESSION_KEY_ACTIVE_PROVIDER] = primary_model_info["provider"]
                st.session_state[SESSION_KEY_ACTIVE_MODEL] = primary_model_info["model"]
                logger.info(f"Quick Setup: Set primary model to free model: {_format_model_id(primary_model_info['provider'], primary_model_info['model'])}")

                # Enable board with available free vision models (limit to e.g., 3)
                board_members_free_vision = free_vision_models[:3]
                if board_members_free_vision:
                    st.session_state[SESSION_KEY_BOARD_ENABLED] = True
                    board_members_dicts = [
                        {"provider": m["provider"], "model": m["model"]}
                        for m in board_members_free_vision
                    ]
                    st.session_state[SESSION_KEY_BOARD_MEMBERS] = board_members_dicts
                    # Set chairperson to the first board member
                    st.session_state[SESSION_KEY_CHAIRPERSON] = board_members_dicts[0]
                    # Set default tasks
                    st.session_state[SESSION_KEY_BOARD_TASKS] = ["viral_analysis", "monetization_analysis"]
                    logger.info(f"Quick Setup: Enabled AI Board with free members: {board_members_dicts}, Chairperson: {board_members_dicts[0]}")
                else:
                    # If no free vision models, disable the board
                    st.session_state[SESSION_KEY_BOARD_ENABLED] = False
                    st.session_state[SESSION_KEY_BOARD_MEMBERS] = []
                    st.session_state[SESSION_KEY_CHAIRPERSON] = None
                    st.session_state[SESSION_KEY_BOARD_TASKS] = [] # Clear tasks too
                    logger.info("Quick Setup: No free vision models found, disabling AI Board.")
                    st.sidebar.info("No free vision models available to enable AI Board.")

                st.success("Configured to use free models.")
                st.rerun() # Rerun to reflect changes in the UI immediately
            else:
                st.sidebar.error("Quick Setup Error: Could not identify a suitable primary free model.")

        else:
            st.sidebar.warning("Quick Setup: No free models seem to be available in the registry.")

    # --- Return Final Configuration ---
    # Construct the return dictionary based on the *current* session state
    final_board_config = {
        "board_enabled": st.session_state[SESSION_KEY_BOARD_ENABLED],
        "board_members": st.session_state[SESSION_KEY_BOARD_MEMBERS],
        "chairperson": st.session_state[SESSION_KEY_CHAIRPERSON],
        "board_tasks": st.session_state[SESSION_KEY_BOARD_TASKS],
    }
    final_primary_config = {
         "provider": st.session_state.get(SESSION_KEY_ACTIVE_PROVIDER),
         "model": st.session_state.get(SESSION_KEY_ACTIVE_MODEL)
    }

    # Ensure primary config reflects potential lack of selection if no providers exist
    if not available_providers:
        final_primary_config = {"provider": None, "model": None}


    final_config = {"primary": final_primary_config, **final_board_config}
    # logger.debug(f"Returning final config: {final_config}")
    return final_config


def create_api_key_management_ui(
    key_manager: APIKeyManager, model_registry: ModelRegistry
):
    """
    Renders UI for managing API keys for AI providers on a dedicated page.

    Args:
        key_manager: An instance of APIKeyManager to handle key storage/retrieval.
        model_registry: An instance of ModelRegistry to identify providers needing keys.
    """
    st.title("API Key Management")
    st.markdown(
        "Manage API keys for different AI providers. Keys are encrypted and stored "
        "locally in `~/.openclip/api_keys.json`."
    )
    # Security warning about hardcoded key
    st.warning(
        "**Security Alert:** A hardcoded encryption key might be used if not configured "
        "properly. **This is insecure for production or shared environments.** "
        "Ensure secure key management (e.g., environment variables, secrets manager) "
        "for real-world deployment."
    )
    # Check if encryption is actually available
    if not fernet:
        st.error(
            "API Key encryption is unavailable. Cannot manage keys securely. "
            "Please ensure the 'cryptography' library is installed and Fernet "
            "initialization was successful (check application logs)."
        )
        return # Stop rendering if encryption isn't working

    # Identify providers that might require an API key based on model definitions
    providers_requiring_keys = sorted(
        {
            prov
            for prov, models in model_registry.KNOWN_MODELS.items()
            # Check if *any* model listed for this provider requires a key
            if any(info.get("requires_api_key", True) for info in models.values()) # Default to True if unspecified
        }
    )

    if not providers_requiring_keys:
        st.info(
            "No configured AI providers appear to require API keys based on the "
            "current Model Registry."
        )
        st.caption(
            "Providers like Ollama typically run locally and do not need API keys."
        )
        return

    # Add note about environment variables overriding saved keys
    st.info(
        "API keys can also be set via environment variables (e.g., `OPENAI_API_KEY`, "
        "`GEMINI_API_KEY`, etc.). Environment variables take precedence over keys saved here."
    )
    st.markdown("---")

    # Create UI sections for each provider needing a key
    for provider in providers_requiring_keys:
        provider_display_name = provider.capitalize()
        st.subheader(f"{provider_display_name} API Key")

        # Check if a key is currently set (via env, file, or session cache)
        # Note: get_key() returns the effective key considering precedence.
        current_key = key_manager.get_key(provider)
        is_key_from_env = key_manager.is_key_from_env(provider) # Check source

        # Special handling for Ollama (typically keyless)
        if provider.lower() == "ollama":
            st.info(
                f"{provider_display_name} typically runs locally and does not require an API key."
            )
            st.caption(
                "Ensure your Ollama server is running and accessible if using Ollama models. "
                "No key needs to be saved here."
            )
        # Handle providers where a key is currently set
        elif current_key:
            if is_key_from_env:
                 st.success(f"Using API key from environment variable for {provider_display_name}.")
                 # Optionally show masked key from env var if needed, but might be less secure
                 # masked_key = "****" + current_key[-4:] if len(current_key) > 4 else "****"
                 # st.caption(f"Key ends in: {masked_key}")
                 st.caption("Remove the environment variable to manage the key via this UI.")
            else:
                # Key is set via the saved file
                masked_key = "****" + current_key[-4:] if len(current_key) > 4 else "****"
                st.success(f"API Key saved (ending in {masked_key}).")

                # Allow deleting the saved key
                if st.button(f"Delete Saved Key for {provider_display_name}", key=f"delete_{provider}_key"):
                    if key_manager.delete_key(provider):
                        st.success(f"Saved API Key for {provider_display_name} deleted.")
                        st.rerun()
                    else:
                        st.error(f"Failed to delete saved key for {provider_display_name}. Check logs.")

                # Allow updating the saved key
                with st.expander(f"Update Saved API Key for {provider_display_name}"):
                    new_key_update = st.text_input(
                        f"Enter New {provider_display_name} Key",
                        type="password",
                        key=f"update_input_{provider}",
                    )
                    if st.button("Save Updated Key", key=f"update_btn_{provider}"):
                        if new_key_update:
                            if key_manager.save_key(provider, new_key_update):
                                st.success(f"API key for {provider_display_name} updated successfully!")
                                st.rerun()
                            else:
                                st.error("Failed to save updated API key. Check logs.")
                        else:
                            st.warning("Please enter the new API key.")

        # Handle providers where no key is currently set (neither env nor file)
        else:
            st.warning(f"API Key for {provider_display_name} is not set.")
            new_key_save = st.text_input(
                f"Enter {provider_display_name} Key",
                type="password",
                key=f"input_{provider}",
                help=f"Paste your API key for {provider_display_name} here."
            )
            if st.button(f"Save Key for {provider_display_name}", key=f"save_btn_{provider}"):
                if new_key_save:
                    # Basic validation (can be enhanced)
                    if len(new_key_save.strip()) < 10:
                        st.warning("API Key seems short. Please verify and save again if correct.")
                    elif key_manager.save_key(provider, new_key_save):
                        st.success(f"API key for {provider_display_name} saved successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to save API key. Check logs.")
                else:
                    st.warning("Please enter an API key.")

        st.markdown("---")

    st.caption(
        "Saved keys are encrypted and stored locally. Environment variables, if set, "
        "always override saved keys."
    )
