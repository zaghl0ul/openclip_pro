from pathlib import Path

PROJECT_DIR = Path("C:/Users/blood/Desktop/interactive_finalizer_agent/openclip_pro")

def clean_markdown_fences():
    files = list(PROJECT_DIR.rglob("*.py"))

    for file_path in files:
        text = file_path.read_text(encoding='utf-8')
        cleaned_text = text.replace("", "").replace("", "")

        if text != cleaned_text:
            file_path.write_text(cleaned_text, encoding='utf-8')
            print(f"[✓] Cleaned: {file_path}")

if __name__ == "__main__":
    clean_markdown_fences()
