from __future__ import annotations
import argparse
from rempf.cli.match import add_match_subparser
from rempf.cli.tsp import add_tsp_subparser

def main() -> None:
    parser = argparse.ArgumentParser(prog="rempf")
    sub = parser.add_subparsers(dest="cmd", required=True)
    add_match_subparser(sub)
    add_tsp_subparser(sub)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
