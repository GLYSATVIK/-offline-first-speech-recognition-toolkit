"""``chunkasr manifest`` — validate, summarise and filter dataset manifests."""

from __future__ import annotations

import argparse

from ..data.manifest import filter_by_duration, manifest_stats, read_manifest, write_manifest
from ..exceptions import ManifestError


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("manifest", help="inspect and filter manifests")
    parser.add_argument("action", choices=["validate", "stats", "filter"])
    parser.add_argument("path", help="path to a JSONL manifest")
    parser.add_argument("--min-duration", type=float, default=0.0, dest="min_duration")
    parser.add_argument("--max-duration", type=float, default=float("inf"), dest="max_duration")
    parser.add_argument("--out", default=None, help="output path for 'filter'")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    try:
        entries = read_manifest(args.path)
    except ManifestError as exc:
        print(f"invalid manifest: {exc}")
        return 1

    if args.action == "validate":
        print(f"ok: {len(entries)} valid entries")
        return 0
    if args.action == "stats":
        for key, value in manifest_stats(entries).items():
            print(f"{key:16s}: {value:.3f}")
        return 0

    kept = filter_by_duration(entries, args.min_duration, args.max_duration)
    if not args.out:
        print(f"error: --out is required for filter (would keep {len(kept)} entries)")
        return 2
    written = write_manifest(args.out, kept)
    print(f"wrote {written} entries to {args.out}")
    return 0
