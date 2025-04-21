# Contains UI generation functions related to AI configuration
import logging
from typing import Dict

import streamlit as st

# Assuming logger is configured elsewhere in the application
# Example: logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Assuming Fernet is imported and initialized elsewhere if encryption is used.
# from cryptography.fernet import Fernet
# Example: fernet = Fernet(ENCRYPTION_KEY)
# If fernet is None, key management UI will show an error.
# This dependency should be handled during application setup.
fernet = None # Placeholder: Replace with actual Fernet instance if available

# Import specific classes needed
# Ensure correct import paths relative to the project structure
from ai.ai_models import APIKeyManager, ModelRegistry

# --- Constants for Session State ---
# Using a prefix helps avoid collisions with other parts of the app
SESSION_KEY_AI_UI_PREFIX = "ai_ui_state_"
SESSION_KEY_ACTIVE_PROVIDER = SESSION_KEY_AI_UI_PREFIX + "active_provider"
SESSION_KEY_ACTIVE_MODEL = SESSION_KEY_AI_UI_PREFIX + "active_model"
SESSION_KEY_BOARD_ENABLED = SESSION_KEY_AI_UI_PREFIX + "board_enabled"
SESSION_KEY_BOARD_MEMBERS = SESSION_KEY_AI_UI_PREFIX + "board_members"
SESSION_KEY_BOARD_TASKS = SESSION_KEY_AI_UI_PREFIX + "board_tasks"
SESSION_KEY_CHAIRPERSON = SESSION_KEY_AI_UI_PREFIX + "chairperson"

# --- Helper Functions ---

def _initialize_session_state_defaults(model_registry: ModelRegistry):
    """Initialize session state keys with default values if they don't exist."""
    if SESSION_KEY_ACTIVE_PROVIDER not in st.session_state:
        providers = model_registry.list_providers()
        st.session_state[SESSION_KEY_ACTIVE_PROVIDER] = providers[0] if providers else None

    if SESSION_KEY_ACTIVE_MODEL not in st.session_state:
        provider = st.session_state.get(SESSION_KEY_ACTIVE_PROVIDER)
        if provider:
            models = model_registry.list_models_for_provider(provider)
            st.session_state[SESSION_KEY_ACTIVE_MODEL] = models[0] if models else None
        else:
            st.session_state[SESSION_KEY_ACTIVE_MODEL] = None

    # Initialize board defaults
    st.session_state.setdefault(SESSION_KEY_BOARD_ENABLED, False)
    st.session_state.setdefault(SESSION_KEY_BOARD_MEMBERS, [])
    st.session_state.setdefault(SESSION_KEY_BOARD_TASKS, ["viral_analysis", "monetization_analysis"])
    st.session_state.setdefault(SESSION_KEY_CHAIRPERSON, None)


def _get_available_board_models(
    model_registry: ModelRegistry, key_manager: APIKeyManager
) -> list[dict]:
    """
    Filter models suitable for the AI Board.
    Requires vision capability and checks API key availability if needed.
    """
    available_models = []
    providers_with_keys_lower = {p.lower() for p in key_manager.list_providers_with_keys()}
    vision_models = model_registry.list_vision_models() # Assumes this returns list of dicts

    for model_info in vision_models:
        provider = model_info.get("provider")
        model_name = model_info.get("model")
        requires_key = model_info.get("requires_api_key", True)

        if not provider or not model_name:
            logger.warning(f"Skipping invalid vision model entry: {model_info}")
            continue

        # Include if the model doesn't require a key, OR if it requires a key AND the key is present.
        if not requires_key or provider.lower() in providers_with_keys_lower:
            available_models.append({"provider": provider, "model": model_name})
        else:
            logger.debug(
                f"Skipping board candidate {provider}:{model_name} - API key required but not found."
            )
    return available_models

# --- Main UI Functions ---

def create_model_selection_ui(
    model_registry: ModelRegistry, key_manager: APIKeyManager
) -> Dict:
    """
    Render UI for selecting AI models and board configuration in Streamlit sidebar.

    Uses st.session_state to persist user selections across reruns.

    Args:
        model_registry: An instance of ModelRegistry containing available models.
        key_manager: An instance of APIKeyManager for checking API key status.

    Returns:
        Dictionary containing the selected primary model config and board config.
        Note: The primary source of truth for selections is st.session_state.
              This return value provides a snapshot of the current config.
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧠 AI Model Configuration")

    # Initialize session state with defaults if necessary
    _initialize_session_state_defaults(model_registry)

    # --- Primary Model Selection ---
    st.sidebar.subheader("Primary Analysis Model")
    providers = model_registry.list_providers()
    if not providers:
        st.sidebar.warning("No AI providers available. Check model registry/dependencies.")
        return {"primary": {"provider": None, "model": None}, "board_enabled": False} # Return empty config

    # Get current provider selection from session state, ensuring it's valid
    current_provider = st.session_state[SESSION_KEY_ACTIVE_PROVIDER]
    if current_provider not in providers:
        current_provider = providers[0] # Fallback to first available
        st.session_state[SESSION_KEY_ACTIVE_PROVIDER] = current_provider # Update state

    provider_index = providers.index(current_provider)

    # Provider selectbox - updates session state on change via the key
    selected_provider = st.sidebar.selectbox(
        "Select AI Provider",
        providers,
        index=provider_index,
        key=SESSION_KEY_ACTIVE_PROVIDER, # Bind selectbox directly to session state key
        # on_change callback could be used here for more complex side effects if needed
    )

    # Get models for the *currently selected* provider
    models = model_registry.list_models_for_provider(selected_provider)
    primary_config = {"provider": selected_provider, "model": None} # Default config

    if not models:
        st.sidebar.warning(f"No models found for provider '{selected_provider}'. Check registry.")
    else:
        # Get current model selection from session state, ensuring it's valid for the selected provider
        current_model = st.session_state[SESSION_KEY_ACTIVE_MODEL]
        if current_model not in models:
            current_model = models[0] # Fallback to first model of the selected provider
            st.session_state[SESSION_KEY_ACTIVE_MODEL] = current_model # Update state

        model_index = models.index(current_model)

        # Model selectbox - updates session state on change via the key
        selected_model = st.sidebar.selectbox(
            "Select Model",
            models,
            index=model_index,
            key=SESSION_KEY_ACTIVE_MODEL, # Bind selectbox directly to session state key
        )
        primary_config["model"] = selected_model

        # Display model info and API key status
        model_info = model_registry.get_model_info(selected_provider, selected_model)
        if model_info:
            cap_str = ", ".join(model_info.get("capabilities", [])) or "N/A"
            st.sidebar.caption(
                f"Type: {model_info.get('type', 'N/A')}, Quality: {model_info.get('quality', 'N/A')}"
            )
            st.sidebar.caption(f"Capabilities: {cap_str}")

            # Check API key requirement and availability
            if model_info.get("requires_api_key", True):
                if not key_manager.get_key(selected_provider):
                    st.sidebar.error(f"⚠️ API key needed for {selected_provider.capitalize()}!")
                    st.sidebar.caption("Add key in '🔑 API Keys' section.")
                else:
                    st.sidebar.success(f"✅ API key found for {selected_provider.capitalize()}.")
            else:
                st.sidebar.info(f"🔑 {selected_provider.capitalize()} model likely doesn't require an API key.")

    # --- Board of Directors Configuration ---
    st.sidebar.subheader("AI Board of Directors")

    # Board enable toggle - updates session state on change via the key
    enable_board = st.sidebar.toggle(
        "Enable AI Board",
        key=SESSION_KEY_BOARD_ENABLED, # Bind toggle directly to session state key
        help="Use multiple AI models (the 'Board') to perform specialized analysis tasks.",
    )

    # Prepare board_config dictionary based on session state
    # Note: Board members, chairperson, tasks are modified *within* the 'if enable_board' block
    # but read directly from session_state later when constructing the final return dict.
    board_config = {"board_enabled": enable_board}

    if enable_board:
        st.sidebar.markdown("Configure the board:")

        # Determine which models can be used as board members
        available_board_models_dicts = _get_available_board_models(model_registry, key_manager)

        if not available_board_models_dicts:
            st.sidebar.warning("No suitable models available for the board (check vision capability & API keys).")
            # Reset board state if no models are available
            st.session_state[SESSION_KEY_BOARD_MEMBERS] = []
            st.session_state[SESSION_KEY_CHAIRPERSON] = None
        else:
            # Format available models for the multiselect widget (provider:model)
            available_board_options_str = sorted([
                f"{m['provider']}:{m['model']}" for m in available_board_models_dicts
            ])

            # Get current selection from session state and format as strings
            current_board_members_dicts = st.session_state[SESSION_KEY_BOARD_MEMBERS]
            current_board_members_str = [
                f"{m['provider']}:{m['model']}" for m in current_board_members_dicts
            ]
            # Filter default selection to only include currently available models
            valid_default_members_str = [
                s for s in current_board_members_str if s in available_board_options_str
            ]

            # Board Members multiselect
            selected_member_strings = st.sidebar.multiselect(
                "Board Members (Vision Models)",
                options=available_board_options_str,
                default=valid_default_members_str,
                key="board_members_multiselect_widget", # Use a distinct key for the widget itself
                help="Select AI models with vision capabilities to act as board members.",
            )

            # Convert selected strings back to list of dicts and save to session state
            updated_board_members = []
            for item_str in selected_member_strings:
                try:
                    prov, mod = item_str.split(":", 1)
                    updated_board_members.append({"provider": prov, "model": mod})
                except ValueError:
                    logger.warning(f"Invalid model format skipped in board selection: {item_str}")
            st.session_state[SESSION_KEY_BOARD_MEMBERS] = updated_board_members

            # --- Advanced Board Settings Expander ---
            with st.sidebar.expander("Advanced Board Settings"):
                # Chairperson Selection
                current_board_members = st.session_state[SESSION_KEY_BOARD_MEMBERS] # Use updated list
                chairperson_options_str = [f"{m['provider']}:{m['model']}" for m in current_board_members]

                if chairperson_options_str:
                    # Get current chairperson from session state, format as string for comparison
                    current_chair_dict = st.session_state[SESSION_KEY_CHAIRPERSON]
                    current_chair_str = None
                    if current_chair_dict:
                        current_chair_str = f"{current_chair_dict['provider']}:{current_chair_dict['model']}"

                    # Ensure the current chairperson is still a valid option (i.e., selected member)
                    if current_chair_str not in chairperson_options_str:
                        current_chair_str = None # Reset if invalid
                        st.session_state[SESSION_KEY_CHAIRPERSON] = None # Update state

                    # Determine index for selectbox, default to 0 if no valid chair selected
                    try:
                        chair_index = chairperson_options_str.index(current_chair_str) if current_chair_str else 0
                    except ValueError: # Should not happen if logic above is correct, but belts and suspenders
                        chair_index = 0
                        st.session_state[SESSION_KEY_CHAIRPERSON] = None # Reset state

                    selected_chair_str = st.selectbox(
                        "Chairperson (Synthesizes Results)",
                        options=chairperson_options_str,
                        index=chair_index,
                        key="chairperson_select_widget", # Distinct widget key
                        help="Select one board member to synthesize the findings of all members.",
                    )

                    # Convert selected string back to dict and save to session state
                    prov, mod = selected_chair_str.split(":", 1)
                    updated_chairperson = {"provider": prov, "model": mod}
                    st.session_state[SESSION_KEY_CHAIRPERSON] = updated_chairperson
                else:
                    st.caption("Select board members first to choose a chairperson.")
                    st.session_state[SESSION_KEY_CHAIRPERSON] = None # Ensure state is cleared

                # Task Selection
                available_board_task_types = ["viral_analysis", "monetization_analysis"] # Add future tasks here
                current_tasks = st.session_state[SESSION_KEY_BOARD_TASKS]
                # Filter default tasks to ensure they are still valid task types
                valid_default_tasks = [t for t in current_tasks if t in available_board_task_types]

                selected_tasks = st.multiselect(
                    "Board Tasks",
                    options=available_board_task_types,
                    default=valid_default_tasks,
                    key=SESSION_KEY_BOARD_TASKS, # Bind directly to session state key
                    help="Select the specific analyses the AI Board should perform.",
                )
                # Session state is updated automatically by the widget via its key

    else:
        # If board is disabled, ensure related session state values are cleared/reset
        # Although not strictly necessary if UI prevents access, it keeps state consistent.
        # st.session_state[SESSION_KEY_BOARD_MEMBERS] = [] # Optional: clear members if disabled
        # st.session_state[SESSION_KEY_CHAIRPERSON] = None # Optional: clear chairperson if disabled
        pass # No UI elements needed if board is disabled


    # --- Quick Setup Button ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("🚀 Quick Setup")
    st.sidebar.caption("Easily configure AI settings with one click.")

    if st.sidebar.button("Use Only Free Models", key="use_free_models_button"):
        # Get list of free models (assuming this returns list of dicts with model info)
        free_models_list = model_registry.list_free_models()
        if free_models_list:
            # Separate vision models
            free_vision_models = [
                m for m in free_models_list
                if "image" in m.get("capabilities", []) and m.get("provider") and m.get("model")
            ]

            primary_model_info = None
            if free_vision_models:
                # Prefer highest quality free vision model for primary
                # Assuming quality is represented as strings like 'premium', 'standard', 'fast'
                # This sort order might need adjustment based on actual quality values
                quality_order = {'premium': 3, 'standard': 2, 'fast': 1}
                free_vision_models.sort(
                    key=lambda x: quality_order.get(x.get("quality", "standard").lower(), 0),
                    reverse=True
                )
                primary_model_info = free_vision_models[0]
            elif free_models_list:
                # Fallback to the first available free model if no vision models
                primary_model_info = free_models_list[0]

            if primary_model_info:
                # Set primary model state
                st.session_state[SESSION_KEY_ACTIVE_PROVIDER] = primary_model_info["provider"]
                st.session_state[SESSION_KEY_ACTIVE_MODEL] = primary_model_info["model"]
                logger.info(f"Quick Setup: Set primary to {primary_model_info['provider']}:{primary_model_info['model']}")

                # Configure board with free vision models (up to 3)
                board_members_to_set = free_vision_models[:3]
                if board_members_to_set:
                    st.session_state[SESSION_KEY_BOARD_ENABLED] = True
                    board_members_dicts = [
                        {"provider": m["provider"], "model": m["model"]} for m in board_members_to_set
                    ]
                    st.session_state[SESSION_KEY_BOARD_MEMBERS] = board_members_dicts
                    # Set chairperson to the first member
                    st.session_state[SESSION_KEY_CHAIRPERSON] = board_members_dicts[0]
                    # Set default tasks
                    st.session_state[SESSION_KEY_BOARD_TASKS] = ["viral_analysis", "monetization_analysis"]
                    logger.info(f"Quick Setup: Enabled board with members: {board_members_dicts}")
                else:
                    # Disable board if no free vision models found
                    st.session_state[SESSION_KEY_BOARD_ENABLED] = False
                    st.session_state[SESSION_KEY_BOARD_MEMBERS] = []
                    st.session_state[SESSION_KEY_CHAIRPERSON] = None
                    st.sidebar.info("Quick Setup: No free vision models available for the AI Board.")

                st.success("Quick Setup: Configured free models.")
                st.rerun() # Rerun to reflect changes in the UI immediately
            else:
                 st.sidebar.error("Quick Setup: Failed to find a suitable primary free model.")

        else:
            st.sidebar.warning("Quick Setup: No free models found in the registry.")


    # --- Return Current Configuration ---
    # This dictionary reflects the current state managed by st.session_state.
    final_config = {
        "primary": primary_config, # Contains provider and model selected above
        "board_enabled": st.session_state[SESSION_KEY_BOARD_ENABLED],
        "board_members": st.session_state[SESSION_KEY_BOARD_MEMBERS],
        "chairperson": st.session_state[SESSION_KEY_CHAIRPERSON],
        "board_tasks": st.session_state[SESSION_KEY_BOARD_TASKS],
    }
    return final_config


def create_api_key_management_ui(
    key_manager: APIKeyManager, model_registry: ModelRegistry
):
    """
    Render UI for API key management on a dedicated Streamlit page.

    Allows users to add, view status of, update, and delete API keys.
    Keys are managed by the APIKeyManager instance.

    Args:
        key_manager: An instance of APIKeyManager to handle key operations.
        model_registry: An instance of ModelRegistry to identify providers needing keys.
    """
    st.title("🔑 API Key Management")
    st.markdown(
        "Manage API keys for different AI providers. Keys are stored locally using encryption."
    )

    # Check if Fernet is available (required for encryption)
    # Assumes 'fernet' is initialized elsewhere and passed or available globally.
    if not fernet:
        st.error("🔒 API Key encryption is unavailable. Cannot manage keys securely.")
        st.warning(
            "Please ensure the 'cryptography' library is installed and Fernet "
            "encryption was successfully initialized (check application logs)."
        )
        # Optionally display instructions on setting environment variables as an alternative
        st.info(
            "Alternatively, you can provide keys via environment variables "
            "(e.g., `OPENAI_API_KEY`, `GEMINI_API_KEY`), which will take precedence."
        )
        return # Stop rendering the rest of the UI if encryption isn't working

    # Security warning about default encryption key (if applicable)
    # TODO: Check if a default/insecure key is being used and display warning conditionally
    st.warning(
        "⚠️ Security Notice: If using the default encryption key, it's **not secure** "
        "for production or shared environments. Ensure a unique, securely managed "
        "encryption key (e.g., via environment variable) is configured for real-world use."
    )

    # Identify providers that actually require an API key based on their models
    providers_requiring_keys = sorted(
        {
            info["provider"]
            for provider_models in model_registry.KNOWN_MODELS.values()
            for info in provider_models.values()
            if info.get("requires_api_key", False) # Check the flag for each model
        }
    )

    if not providers_requiring_keys:
        st.info(
            "No providers currently configured in the Model Registry require API keys."
        )
        st.caption("Providers like local Ollama typically do not need API keys.")
        return

    # Note about environment variables overriding saved keys
    st.info(
        "🔑 API keys can also be set via environment variables "
        "(e.g., `OPENAI_API_KEY`, `GEMINI_API_KEY`). Environment variables "
        "override keys saved through this interface."
    )
    st.markdown("---")

    # Display input/status for each provider requiring a key
    for provider in providers_requiring_keys:
        st.subheader(f"{provider.capitalize()} API Key")

        # Get the current key status from the key manager
        # key_manager.get_key() typically checks env vars first, then saved keys.
        current_key = key_manager.get_key(provider)
        key_source = key_manager.get_key_source(provider) # 'environment', 'file', 'none'

        if current_key:
            # Display masked key and source
            masked_key = (
                f"{'*' * (len(current_key) - 4)}{current_key[-4:]}"
                if len(current_key) > 4
                else "****"
            )
            if key_source == "environment":
                st.success(f"✅ Key found via environment variable (ending in {masked_key}).")
                st.caption("Environment variable takes precedence. To manage via UI, unset the variable.")
            elif key_source == "file":
                st.success(f"✅ Key loaded from local storage (ending in {masked_key}).")

                # Allow deleting the saved key (won't affect env var)
                if st.button(f"Delete Saved Key for {provider.capitalize()}", key=f"delete_{provider}_key"):
                    if key_manager.delete_key(provider):
                        st.success(f"Saved API Key for {provider.capitalize()} deleted.")
                        st.rerun()
                    else:
                        st.error(f"Failed to delete saved key for {provider.capitalize()}.")

                # Allow updating the saved key (won't affect env var)
                with st.expander(f"Update Saved API Key for {provider.capitalize()}"):
                    new_key_update = st.text_input(
                        f"Enter New {provider.capitalize()} Key",
                        type="password",
                        key=f"update_input_{provider}",
                        help="Enter the complete new API key here.",
                    )
                    if st.button("Save Updated Key", key=f"update_btn_{provider}"):
                        if new_key_update:
                            if key_manager.save_key(provider, new_key_update):
                                st.success(f"API key for {provider.capitalize()} updated successfully!")
                                st.rerun()
                            else:
                                st.error("Failed to save updated API key (check logs).")
                        else:
                            st.warning("Please enter the new API key.")
            else: # Should not happen if current_key is true, but handle defensively
                 st.info(f"Key found for {provider.capitalize()} (source unknown).")

        else:
            # API key is not set (neither env var nor file)
            st.warning(f"API Key for {provider.capitalize()} is not set.")
            new_key_save = st.text_input(
                f"Enter {provider.capitalize()} Key to Save",
                type="password",
                key=f"input_{provider}",
                help="Enter the API key to save it locally (encrypted).",
            )
            if st.button(f"Save Key for {provider.capitalize()}", key=f"save_btn_{provider}"):
                if new_key_save:
                    # Basic validation example (adjust length as needed)
                    if len(new_key_save.strip()) < 10:
                        st.warning("API Key seems short. Please double-check.")
                    elif key_manager.save_key(provider, new_key_save):
                        st.success(f"API key for {provider.capitalize()} saved successfully!")
                        st.rerun() # Update UI to show the key is set
                    else:
                        st.error("Failed to save API key (check logs).")
                else:
                    st.warning("Please enter an API key.")

        st.markdown("---") # Separator between providers

    # Final note on storage location and security
    st.markdown(
        f"""
        **Key Storage:** Keys saved via this UI are encrypted (if encryption is available)
        and stored locally, typically in a file like `~/.<your_app_name>/api_keys.json`.
        The exact path depends on the `APIKeyManager` implementation.

        **Security:** Remember that environment variables take precedence. The security of
        saved keys depends heavily on the secure management of the encryption key used by Fernet.
        """
    )
