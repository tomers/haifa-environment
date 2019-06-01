import click
from src.flares.flare_reports import run_all


@click.group()
def flares():
    """Flares commands"""


@flares.command()
def run():
    run_all()
