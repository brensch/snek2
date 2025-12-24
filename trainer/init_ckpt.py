import argparse
import os

import torch

from model import EgoSnakeNet


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="models/latest.pt")
    p.add_argument("--in-channels", type=int, default=10)
    p.add_argument("--width", type=int, default=11)
    p.add_argument("--height", type=int, default=11)
    args = p.parse_args()

    out_path = str(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    model = EgoSnakeNet(width=int(args.width), height=int(args.height), in_channels=int(args.in_channels))
    torch.save(model.state_dict(), out_path)
    print(f"Wrote random checkpoint: {out_path} (in_channels={args.in_channels})")


if __name__ == "__main__":
    main()
