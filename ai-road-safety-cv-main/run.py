import subprocess
from pathlib import Path


def launch_application():
    """Start the Streamlit road-safety application."""

    project_root = Path(__file__).resolve().parent
    streamlit_app = project_root / "app" / "app.py"

    subprocess.run(
        ["streamlit", "run", str(streamlit_app)],
        check=True
    )


if __name__ == "__main__":
    launch_application()
