import sys


def main() -> None:
    print("SpaceCart console app")
    print("Type 'help' for commands, 'quit' to exit.")
    while True:
        try:
            command = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if command in ("quit", "exit"):
            break
        if command == "help":
            print("Available commands: help, quit")
        elif command == "":
            continue
        else:
            print(f"Unknown command: {command}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
