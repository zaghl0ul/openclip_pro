
# Contains UI generation functions related to AI configuration
import streamlit as st
from typing import Dict  # Used in type hint

# Import specific classes needed
from ai.ai_models import ModelRegistry, APIKeyManager

# --- Constants for Session State ---
# Using a prefix helps avoid collisions with other parts of the application state
SESSION_KEY_AI_UI_PREFIX = "ai_ui_state_"
SESSION_KEY_ACTIVE_PROVIDER = SESSION_KEY_AI_UI_PREFIX + "active_provider"
SESSION_KEY_ACTIVE_MODEL = SESSION_KEY_AI_UI_PREFIX + "active_model"
# Board-related keys (kept for potential future use or compatibility)
SESSION_KEY_BOARD_ENABLED = SESSION_KEY_AI_UI_PREFIX + "board_enabled"
SESSION_KEY_BOARD_MEMBERS = SESSION_KEY_AI_UI_PREFIX + "board_members"
SESSION_KEY_BOARD_TASKS = SESSION_KEY_AI_UI_PREFIX + "board_tasks"
SESSION_KEY_CHAIRPERSON = SESSION_KEY_AI_UI_PREFIX + "chairperson"


def initialize_ai_session_state(model_registry: ModelRegistry):
    """
    Initializes session state variables related to AI configuration.

    Sets default values for provider/model selection and board settings
    if they don't already exist in st.session_state. This ensures other
    parts of the app can safely access these state keys.

    Args:
        model_registry: An instance of ModelRegistry used to fetch
                        default provider and model lists.
    """
    # --- Initialize Board-related State (if not present) ---
    # These are kept for potential future use or backward compatibility.
    defaults = {
        SESSION_KEY_BOARD_ENABLED: False,
        SESSION_KEY_BOARD_MEMBERS: [],
        SESSION_KEY_BOARD_TASKS: ["viral_analysis", "monetization_analysis"],
        SESSION_KEY_CHAIRPERSON: None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    # --- Initialize Model Selection State (if not present) ---
    if SESSION_KEY_ACTIVE_PROVIDER not in st.session_state:
        providers = model_registry.list_providers()
        # Set the first available provider as default, or None if no providers
        st.session_state[SESSION_KEY_ACTIVE_PROVIDER] = providers[0] if providers else None

    if SESSION_KEY_ACTIVE_MODEL not in st.session_state:
        provider = st.session_state.get(SESSION_KEY_ACTIVE_PROVIDER)
        if provider:
            models = model_registry.list_models_for_provider(provider)
            # Set the first available model for the selected provider as default
            st.session_state[SESSION_KEY_ACTIVE_MODEL] = models[0] if models else None
        else:
            # Ensure the key exists even if no provider is selected
            st.session_state[SESSION_KEY_ACTIVE_MODEL] = None


def create_model_selection_ui(
    model_registry: ModelRegistry, key_manager: APIKeyManager # Original signature kept, though key_manager unused here
) -> Dict:
    """
    Initializes AI-related session state variables but renders no UI elements.

    This function primarily ensures that default session state values for AI
    provider/model selection and potentially other features (like the 'board')
    are set. It returns an empty dictionary, preserving the original function's
    output signature, even though the UI rendering aspects have been removed
    from this specific function.

    Args:
        model_registry: An instance of ModelRegistry, passed to state initializer.
        key_manager: An instance of APIKeyManager (currently unused in this function).

    Returns:
        An empty dictionary, signifying no specific UI configuration output.
    """
    # Ensure AI-related session state variables are initialized with defaults.
    initialize_ai_session_state(model_registry)

    # No UI is rendered here; return an empty configuration dictionary as per
    # the function's historical behavior or updated design.
    return {}


def create_api_key_management_ui(
    key_manager: APIKeyManager, model_registry: ModelRegistry
):
    """
    Renders the Streamlit UI for managing API keys for various AI providers.

    This UI allows users to:
    - View the status of API keys (set or not set, considering env vars).
    - Add new API keys for supported providers (saved encrypted locally).
    - Update existing saved API keys.
    - Delete saved API keys.

    Args:
        key_manager: An instance of APIKeyManager responsible for secure
                     storage, retrieval, and encryption/decryption of keys.
                     It should handle checking environment variables internally.
        model_registry: An instance of ModelRegistry used to identify which
                        providers might require API keys based on their models.
    """
    st.title("🔑 API Key Management")
    st.markdown(
        "Manage API keys for different AI providers. Keys saved here are encrypted and stored locally."
    )

    # --- Security Warnings and Information ---
    st.warning(
        "**Security Notice:** The encryption used for storing keys locally relies "
        "on an encryption key. **Ensure this encryption key is managed securely** "
        "(e.g., via environment variables, secrets management) and is **not hardcoded** in production."
    )
    st.info(
        "**Key Precedence:** API keys provided via **environment variables** "
        "(e.g., `OPENAI_API_KEY`, `GEMINI_API_KEY`) will **always override** keys saved through this UI."
    )

    # --- Check for Encryption Availability ---
    # Assumes key_manager has a method to check if encryption is properly configured
    if not key_manager.is_encryption_available():
        st.error("API Key encryption mechanism is not available.")
        st.warning(
            "Cannot manage keys securely. Please ensure the `cryptography` library is installed "
            "and the necessary encryption key was loaded successfully during application startup. "
            "Check application logs for details."
        )
        return # Stop rendering the rest of the UI if encryption isn't working

    # --- Identify Providers Requiring Keys ---
    try:
        # Get models known by the registry; handle potential errors if method is missing/fails
        known_models = model_registry.get_known_models()
        if not isinstance(known_models, dict):
             raise TypeError("Expected get_known_models to return a dictionary.")
    except AttributeError:
        st.error("Failed to retrieve known models: `get_known_models` method not found on ModelRegistry.")
        return
    except Exception as e:
        st.error(f"Error retrieving models from ModelRegistry: {e}")
        return

    providers_requiring_keys = sorted(
        prov
        for prov, models in known_models.items()
        # Check if any model listed for the provider explicitly requires an API key
        if isinstance(models, dict) and any(info.get("requires_api_key", False) for info in models.values())
    )

    if not providers_requiring_keys:
        st.info("No configured AI providers were found that explicitly require API keys.")
        st.caption(
            "Providers like Ollama often run locally and might not need API keys, "
            "or key requirements might be managed differently (e.g., specific model configurations)."
        )
        return # Nothing more to display if no providers need keys

    # --- UI Loop for Each Provider ---
    for provider in providers_requiring_keys:
        st.subheader(f"{provider.capitalize()} API Key Management")

        # --- Special Handling for Providers like Ollama ---
        # Adapt this condition if other providers need similar non-standard key handling.
        if provider.lower() == "ollama":
            st.info(
                f"{provider.capitalize()} typically runs as a local service. "
                "It usually doesn't require a single 'API key' like cloud providers. "
                "Connection details (like hostname/port) are often configured via environment variables (`OLLAMA_HOST`) or other settings."
            )
            st.markdown("---")
            continue # Skip standard key input UI for this provider

        # --- Standard API Key Input/Update/Delete Logic ---
        # Use unique keys for Streamlit widgets within the loop to avoid state conflicts
        key_base = f"{provider}_api_key_ui" # More specific base key
        input_key = f"input_{key_base}"
        save_button_key = f"save_{key_base}"
        update_input_key = f"update_input_{key_base}"
        update_button_key = f"update_{key_base}"
        delete_button_key = f"delete_{key_base}"
        update_expander_key = f"expander_{key_base}"

        # Check the current key status using the key manager.
        # IMPORTANT: key_manager.get_key() should internally prioritize environment variables.
        current_key = key_manager.get_key(provider)
        # Check if the key *specifically* comes from a saved file (for delete/update UI)
        is_key_saved_locally = key_manager.is_key_saved_locally(provider) # Assumes this method exists

        if current_key:
            # --- Key is SET (potentially from env var or file) ---
            masked_key = "****" + current_key[-4:] if len(current_key) > 4 else "****"
            source = "environment variable or saved file"
            if key_manager.is_key_from_env(provider): # Assumes this method exists
                source = "environment variable"
            elif is_key_saved_locally:
                source = "saved file"

            st.success(f"API Key for {provider.capitalize()} is configured (ending in {masked_key}).")
            st.caption(f"Source: {source}.")

            # Only show delete/update options if the key is managed via the local file
            if is_key_saved_locally:
                 # Option to Delete Saved Key
                if st.button(f"Delete Saved Key for {provider.capitalize()}", key=delete_button_key):
                    if key_manager.delete_key(provider):
                        st.success(f"Saved API Key for {provider.capitalize()} deleted successfully.")
                        # Clear any session cache and rerun to reflect the change
                        key_manager.clear_session_key(provider) # Assumes this method exists
                        st.rerun()
                    else:
                        st.error(f"Failed to delete saved API key for {provider.capitalize()}. Check application logs.")

                # Option to Update Saved Key
                with st.expander(f"Update Saved API Key for {provider.capitalize()}", key=update_expander_key):
                    new_key_update = st.text_input(
                        "Enter New Key to Save",
                        type="password",
                        key=update_input_key,
                        help=f"Enter the new {provider.capitalize()} key here to replace the currently saved one."
                    )
                    if st.button("Save Updated Key", key=update_button_key):
                        # Basic validation: ensure input is not empty
                        if new_key_update and new_key_update.strip():
                            if key_manager.save_key(provider, new_key_update.strip()):
                                st.success(f"Saved API key for {provider.capitalize()} updated successfully!")
                                key_manager.clear_session_key(provider) # Assumes method exists
                                st.rerun()
                            else:
                                st.error("Failed to save updated API key. Check application logs.")
                        else:
                            st.warning("Please enter the new API key before saving.")
            elif key_manager.is_key_from_env(provider):
                 st.info("This key is provided by an environment variable and cannot be modified here.")

        else:
            # --- Key is NOT SET (neither env var nor saved file) ---
            st.warning(f"API Key for {provider.capitalize()} is not set.")

            # Input field to add a new key (will be saved locally)
            new_key_add = st.text_input(
                f"Enter {provider.capitalize()} API Key to Save",
                type="password",
                key=input_key,
                help=f"Paste your API key for {provider.capitalize()} here. It will be saved encrypted locally."
            )
            if st.button(f"Save Key for {provider.capitalize()}", key=save_button_key):
                 # Basic validation: ensure input is not empty
                if new_key_add and new_key_add.strip():
                    # Optional: Basic length check as a simple heuristic
                    if len(new_key_add.strip()) < 10:
                         # Use info level for gentle warning, doesn't prevent saving
                        st.info("The entered key seems short. Please double-check it, then save again if correct.")
                    # Attempt to save the key
                    elif key_manager.save_key(provider, new_key_add.strip()):
                        st.success(f"API key for {provider.capitalize()} saved successfully!")
                        key_manager.clear_session_key(provider) # Assumes method exists
                        st.rerun()
                    else:
                        st.error("Failed to save API key. Check application logs.")
                else:
                    st.warning("Please enter an API key before saving.")

        st.markdown("---") # Separator after each provider's section

    # --- Final Notes ---
    st.markdown("---")
    st.subheader("Storage Information")
    key_file_path_info = "Unable to determine key file path."
    try:
        # Attempt to get the path if the method exists in APIKeyManager
        if hasattr(key_manager, "get_key_file_path"):
            key_file_path = key_manager.get_key_file_path()
            key_file_path_info = f"Keys saved via this UI are typically stored encrypted in a local file (e.g., `{key_file_path}`)."
        else:
             key_file_path_info = "Key file path determination is not available from APIKeyManager."
    except Exception as e:
         st.warning(f"Could not determine key file path: {e}")
         key_file_path_info = "Error occurred while determining key file path."


    # Display storage info, including provider example if possible
    provider_example = providers_requiring_keys[0].upper() if providers_requiring_keys else "PROVIDER"
    st.caption(
        f"""
        - {key_file_path_info}
        - **Environment variables always take precedence.** If an environment variable like `{provider_example}_API_KEY` is set, it will be used instead of any key saved here.
        - Ensure the encryption method used by the application is secure for your deployment context.
        """
    )
