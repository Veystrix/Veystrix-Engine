"""Create a local expert pack from installed checkpoint shards."""
import argparse
from pathlib import Path

from models.nemotron35.safetensors_index import index_checkpoint
from .expert_pack import build, default_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime", choices=("flashnext", "nemotron35"))
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("destination", type=Path, nargs="?",
                        help="pack file to write; omitted: the default path the engine looks at for this checkpoint")
    parser.add_argument("--layers", type=int, nargs="+", help="pack only these layers for a smaller comparison")
    parser.add_argument("--content-addressed", action="store_true",
                        help="treat destination as an existing directory and name the pack by its SHA-256")
    parser.add_argument("--print-default-path", action="store_true",
                        help="print where the engine looks for this checkpoint's pack and exit")
    args = parser.parse_args()
    try:
        checkpoint = args.checkpoint.expanduser()
        index = index_checkpoint(checkpoint)
        if args.print_default_path:
            print(default_path(checkpoint, index, args.runtime))
            return
        if args.destination is None:
            if args.layers or args.content_addressed:
                parser.error("the default path holds a full pack; name a destination for a partial or hashed one")
            destination = default_path(checkpoint, index, args.runtime)
            if destination.is_file():
                print(f"{destination} already exists")
                return
            destination.parent.mkdir(parents=True, exist_ok=True)
        else:
            destination = args.destination.expanduser()
        result = build(checkpoint, index, args.runtime, destination,
                       layers=args.layers, content_addressed=args.content_addressed)
        destination, size = result if args.content_addressed else (destination, result)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Expert packing failed: {error}\n")
    print(f"Wrote {size:,} bytes to {destination}")


if __name__ == "__main__":
    main()
