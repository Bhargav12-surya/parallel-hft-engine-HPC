"""Save an authentic headless Textual capture for documentation."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from .app import HFTLabApp


async def capture(path: Path, page: str = "home") -> Path:
    app = HFTLabApp()
    async with app.run_test(size=(140, 48)) as pilot:
        await asyncio.wait_for(app.workers.wait_for_complete(), timeout=45)
        app.show_page(page)
        await pilot.pause()
        path = path.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        app.save_screenshot(filename=path.name, path=str(path.parent))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/tui.svg"))
    parser.add_argument("--page", choices=("home", "data", "strategy", "engine", "config", "monitor", "results", "training", "history", "doctor", "help"), default="home")
    args = parser.parse_args()
    print(asyncio.run(capture(args.output, args.page)))


if __name__ == "__main__":
    main()
