#!/usr/bin/env python3

import click
from cli_groups.flares import flares


@click.group()
def cli():
    """Command-line tool for various tools"""
    pass


cli.add_command(flares)


def init():
    cli()


if __name__ in ['__main__']:
    init()
