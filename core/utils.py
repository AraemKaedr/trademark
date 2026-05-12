from pathlib import Path

def get_project_root() -> Path:
    """Возвращает корень проекта"""
    return Path(__file__).parent.parent