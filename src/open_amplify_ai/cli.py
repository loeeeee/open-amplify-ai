import argparse
import sys
import logging

from open_amplify_ai import probe_api, server

logger = logging.getLogger(__name__)


def main() -> None:
    """Entry point for the Amplify AI CLI."""
    parser = argparse.ArgumentParser(
        prog="amplify",
        description="Amplify AI Compatibility Layer CLI"
    )

    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        required=True,
        help="Available subcommands"
    )

    # Probe subcommand
    subparsers.add_parser(
        "probe",
        help="Run the API prober to document endpoints"
    )

    # Server subcommand
    server_parser = subparsers.add_parser(
        "server",
        help="Start the OpenAI-compatible FastAPI server locally"
    )
    server_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to run the server on (default: 8080 or AMPLIFY_SERVER_PORT)"
    )

    args = parser.parse_args()

    if args.command == "probe":
        probe_api.main()
    elif args.command == "server":
        server.run(port=args.port)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
