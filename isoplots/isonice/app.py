import json
import logging
import os

from pathlib import Path

import click
from nicegui import (
    app,
    ui
)

from isoplots.isonice import Config
from isoplots.isonice.tabs import Tabs
from isoplots.isonice.utils import ports


Logger = logging.getLogger(__name__)

Base = Path(__file__).parent.resolve()
Static = Base / 'static'


def root():
    app.add_static_files('/static', str(Static))
    ui.add_head_html('''
        <link rel="stylesheet" href="/static/styles.css">
        <link rel="stylesheet" href="/static/jse-theme-dark.css">
    ''')

    # Makes top-level objects (like Tabs) use the full height of the screen
    ui.context.client.content.classes('h-screen')

    dark = ui.dark_mode()
    dark.enable()

    GUI = Tabs()
    GUI.resetTabs()


def launch(path=".", config=None, check=False, **kwargs):
    """
    Launches the NiceGUI server

    Parameters
    ----------
    path : str, default=None
        Optional path to initialize with
    check : bool, default=False
        Check for an open port, may delay startup
    **kwargs : dict
        Additional key-word arguments passed directly to ui.run
    """
    if check:
        Logger.info("Checking ports")
        ports.checkPorts(kwargs.get("port", 8080))

    # if path:
    #     Logger.info(f"Setting path: {path}")
    #     GUI.tabs["Setup"].search.set_value(path)

    if config is None:
        config = Static / "paths.ini"

    if Path(config).exists():
        Logger.info(f"Reading config: {config}")
        Config.read(config)

    Logger.info("Launching")
    ui.run(root, **kwargs)


@click.command()
@click.option("-p", "--path", default=".", help="Working directory to initially load with")
@click.option("-c", "--config", help="Configuration for isonice")
@click.option("--check", default=False, is_flag=True, help="Check an open port before launching")
@click.option("--host", type=str, help="Starts the server on this host. Defaults to localhost")
@click.option("--port", type=int, default=8080, help="Port to use. Defaults to 8080")
@click.option("-r", "--reload", default=False, is_flag=True, help="Enable hot-reloading")
def cli(**kwargs):
    """
    Launches the NiceGUI browser server
    """
    # Save args to an env var so the forked process can retrieve them
    os.environ["_ISONICE_ARGS"] = json.dumps(kwargs)
    launch(**kwargs)


if __name__ == "__main__":
    logging.basicConfig(level="DEBUG")
    cli()

# NiceGUI spawns the page in a separate process when hot-reloading is enabled
elif __name__ == "__mp_main__":
    logging.basicConfig(level="DEBUG")

    # Retrieve and delete saved arguments
    kwargs = json.loads(os.environ.get("_ISONICE_ARGS", "{}"))
    if "_ISONICE_ARGS" in os.environ:
        del os.environ["_ISONICE_ARGS"]

    launch(**kwargs)
