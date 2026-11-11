"""Interactive HFT-LAB terminal application."""


def run_app() -> None:
    """Launch Textual without importing optional UI dependencies in the CLI."""
    from .app import HFTLabApp

    HFTLabApp().run()


__all__ = ["run_app"]
