import os
import google.generativeai as genai
from pathlib import Path

# === CONFIGURATION ===
PROJECT_DIR = Path("C:/Users/blood/Desktop/interactive_finalizer_agent/openclip_pro")
MODEL_NAME = "gemini-2.5-pro-exp-03-25"
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def list_python_files(directory):
    return [f for f in directory.rglob("*.py") if "__pycache__" not in f.parts]

def select_file(files):
    print("\nAvailable Python files:\n")
    for idx, file in enumerate(files):
        print(f"{idx + 1}. {file.relative_to(PROJECT_DIR)}")

    while True:
        choice = input("\nSelect file number to refactor: ")
        if choice.isdigit() and 1 <= int(choice) <= len(files):
            return files[int(choice) - 1]
        print("Invalid choice. Try again.")

def generate_refactor_prompt(file_path):
    code = file_path.read_text(encoding='utf-8', errors='ignore')
    return f"""You are a senior Python engineer. Refactor this Python code:
- Remove unused imports/variables.
- Improve readability and structure.
- Add helpful comments where needed.
- Don't alter logic unless clearly flawed.

Python code:

{code}

Return only corrected Python code without markdown formatting or explanations.
"""

def send_to_gemini(prompt):
    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(prompt)
    return response.text.strip()

def confirm_and_save(original_code, refactored_code, file_path):
    print("\n===== REFACTORED CODE PREVIEW =====\n")
    print(refactored_code[:2000])  # Preview first 2000 chars

    confirm = input("\nApply refactor to file? (y/n): ").lower().strip()
    if confirm == "y":
        backup_path = file_path.with_suffix('.py.bak')
        file_path.rename(backup_path)
        file_path.write_text(refactored_code, encoding='utf-8')
        print(f"✅ Refactor applied. Original backed up at: {backup_path}")
    else:
        print("❌ Changes discarded.")

def main():
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ GEMINI_API_KEY not set. Exiting.")
        return

    files = list_python_files(PROJECT_DIR)
    if not files:
        print("❌ No Python files found. Exiting.")
        return

    selected_file = select_file(files)
    prompt = generate_refactor_prompt(selected_file)

    try:
        refactored_code = send_to_gemini(prompt)
        original_code = selected_file.read_text(encoding='utf-8', errors='ignore')
        confirm_and_save(original_code, refactored_code, selected_file)
    except Exception as e:
        print(f"❌ Error during refactoring: {e}")

if __name__ == "__main__":
    main()
