import os
import subprocess
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent / "openclip_pro"
EXCLUDE_DIRS = {'__pycache__', 'node_modules', '.git', 'venv', 'env'}
PYTHON_EXT = ('.py',)
EDITOR = os.environ.get("EDITOR", "code")

def list_project_files():
    print("\n[*] Scanning files...\n")
    files = []
    for root, dirs, filenames in os.walk(PROJECT_DIR):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for file in filenames:
            if file.endswith(PYTHON_EXT):
                files.append(Path(root) / file)
    return files

def prompt_action(prompt, default="y"):
    choice = input(f"{prompt} [{'Y/n' if default == 'y' else 'y/N'}]: ").strip().lower()
    return (choice == '' and default == 'y') or choice == 'y'

def lint_code():
    if prompt_action("Run ruff linter?"):
        try:
            subprocess.run(["ruff", "check", str(PROJECT_DIR)], check=False)
        except Exception as e:
            print(f"[!] Ruff error: {e}")

def format_code():
    if prompt_action("Run black formatter?"):
        try:
            subprocess.run(["black", str(PROJECT_DIR)], check=False)
        except Exception as e:
            print(f"[!] Black error: {e}")

def check_dependencies():
    if prompt_action("Run `pip check` for broken dependencies?"):
        try:
            subprocess.run(["pip", "check"], check=False)
        except Exception as e:
            print(f"[!] pip check error: {e}")

def generate_requirements():
    req_file = PROJECT_DIR / "requirements.txt"
    if not req_file.exists() and prompt_action("No requirements.txt found. Generate from current env?"):
        try:
            with open(req_file, "w") as f:
                subprocess.run(["pip", "freeze"], stdout=f)
        except Exception as e:
            print(f"[!] Error generating requirements.txt: {e}")

def write_placeholder_tests():
    tests_dir = PROJECT_DIR / "tests"
    tests_dir.mkdir(exist_ok=True)
    files = list_project_files()
    for file_path in files:
        if "test_" not in file_path.name and "tests" not in file_path.parts:
            test_name = f"test_{file_path.stem}.py"
            test_path = tests_dir / test_name
            if not test_path.exists() and prompt_action(f"Write a placeholder test for {file_path.name}?"):
                try:
                    with open(test_path, "w") as f:
                        f.write(f"""import pytest

def test_placeholder():
    assert True  # TODO: Add tests for {file_path.name}
""")
                except Exception as e:
                    print(f"[!] Failed to create test file {test_path}: {e}")

def finalize_interactive():
    print("\n" + "=" * 50)
    print("        OPENCLIP PRO FINALIZER (INTERACTIVE MODE)        ")
    print("=" * 50 + "\n")

    lint_code()
    format_code()
    check_dependencies()
    generate_requirements()
    write_placeholder_tests()

    print("\n" + "=" * 50)
    print("        FINALIZATION COMPLETE         ")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    finalize_interactive()
