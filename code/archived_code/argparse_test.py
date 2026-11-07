import argparse

# 1. Initialize the parser
parser = argparse.ArgumentParser(description="A simple CLI tool example.")

# 2. Add a positional argument (required by default)
parser.add_argument("name", type=str, help="The name of the user.")

# 3. Add an optional argument (flag)
parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity.")

# 4. Parse the arguments
args = parser.parse_args()

# Use the arguments via dot notation
if args.verbose:
    print(f"Verbose mode activated for: {args.name}")
else:
    print(f"Hello, {args.name}!")
