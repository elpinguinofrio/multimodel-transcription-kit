"""Command line boundary for applications that do not import Python packages."""

import sys

from . import http_client, runner


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "http":
        return http_client.main(argv[1:])
    if argv and argv[0] == "usage":
        from .usage import main as usage_main
        return usage_main(argv[1:])
    return runner.main(argv)
