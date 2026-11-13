"""Create a local expert pack from installed checkpoint shards."""
import argparse
from pathlib import Path

from models.nemotron35.safetensors_index import index_checkpoint
from .expert_pack import build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime", choices=("flashnext", "nemotron35"))
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--layers", type=int, nargs="+", help="pack only these layers for a smaller comparison")
    parser.add_argument("--content-addressed", action="store_true",
                        help="treat destination as an existing directory and name the pack by its SHA-256")
    args = parser.parse_args()
    try:
        checkpoint = args.checkpoint.expanduser()
        destination = args.destination.expanduser()
        result = build(checkpoint, index_checkpoint(checkpoint), args.runtime, destination,
                       layers=args.layers, content_addressed=args.content_addressed)
        destination, size = result if args.content_addressed else (destination, result)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Expert packing failed: {error}\n")
    print(f"Wrote {size:,} bytes to {destination}")


if __name__ == "__main__":
    main()
