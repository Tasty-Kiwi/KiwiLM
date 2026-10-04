"""Export the pinned Dense release for the browser-local playground."""

import argparse
import json
from pathlib import Path

import torch

from kiwilm.browser_export import export_browser_bundle

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--bundle", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--revision", required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(export_browser_bundle(args.bundle, args.output, revision=args.revision),
                     indent=2))
