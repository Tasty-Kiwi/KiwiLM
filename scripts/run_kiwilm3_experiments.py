"""Plan/preflight or explicitly execute bounded CPU experiments B/C; never allocate a VM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kiwilm.data import PreparedTokenData
from kiwilm.v3.experiment_runner import preflight, run_candidate, write_comparisons, write_json
from kiwilm.v3.experiments import CANDIDATES, ExperimentControls, build_suite, validate_suite
from kiwilm.v3.tokenizer import MaskBPETokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "preflight", "train", "compare"))
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument(
        "--qualification",
        action="store_true",
        help="Width16 bounded qualification, not quality evidence",
    )
    parser.add_argument(
        "--controls", type=Path, help="Optional explicit JSON ExperimentControls for planning"
    )
    parser.add_argument("--candidate", choices=CANDIDATES)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--start-training", action="store_true")
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--resume-receipt", type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--reports", nargs="+", type=Path)
    args = parser.parse_args()
    if args.action == "plan":
        if args.data_dir is None or args.tokenizer is None:
            parser.error("plan requires --data-dir and --tokenizer")
        data, tokenizer = PreparedTokenData(args.data_dir), MaskBPETokenizer.load(args.tokenizer)
        tokenizer.assert_base_compatible(data.tokenizer)
        controls = (
            ExperimentControls(**json.loads(args.controls.read_text())) if args.controls else None
        )
        suite = build_suite(
            data_fingerprint=data.fingerprint,
            tokenizer_sha256=tokenizer.fingerprint,
            vocab_size=tokenizer.vocab_size,
            qualification=args.qualification,
            controls=controls,
        )
        for candidate in CANDIDATES:
            preflight(suite, candidate, data, tokenizer)
        write_json(args.suite, suite)
        result = {
            "suite": str(args.suite),
            "identity": suite["identity"],
            "training_started": False,
        }
    else:
        suite = json.loads(args.suite.read_text())
        validate_suite(suite)
        if args.action == "compare":
            if not args.reports or args.output_dir is None:
                parser.error("compare requires --reports and --output-dir")
            reports = {}
            for path in args.reports:
                report = json.loads(path.read_text())
                name = report["provenance"]["candidate"]
                if name in reports:
                    parser.error("duplicate candidate report")
                reports[name] = report
            result = write_comparisons(suite, reports, args.output_dir)
        else:
            if args.data_dir is None or args.tokenizer is None or args.candidate is None:
                parser.error("preflight/train require --data-dir, --tokenizer and --candidate")
            data, tokenizer = (
                PreparedTokenData(args.data_dir),
                MaskBPETokenizer.load(args.tokenizer),
            )
            if args.action == "preflight":
                result = preflight(suite, args.candidate, data, tokenizer)
            else:
                if not args.start_training or args.output_dir is None:
                    parser.error(
                        "train requires explicit --start-training and a candidate --output-dir"
                    )
                result = run_candidate(
                    suite,
                    args.candidate,
                    data,
                    tokenizer,
                    args.output_dir,
                    start_training=True,
                    stop_after_step=args.stop_after_step,
                    resume_receipt=args.resume_receipt,
                    expected_checkpoint_sha256=args.checkpoint_sha256,
                )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
