import argparse

parser = argparse.ArgumentParser(description="A simple CLI tool example.")

parser.add_argument("name", type=str, help="The name of the user.")

parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity.")

args = parser.parse_args()

if args.verbose:
    print(f"Verbose mode activated for: {args.name}")
else:
    print(f"Hello, {args.name}!")
