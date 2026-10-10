"""User-run V3 NVIDIA CLI; no arguments show a non-allocating plan."""

from kiwilm.v3.colab_cli import main

if __name__ == "__main__":
    raise SystemExit(main(default_gpu="L4"))
