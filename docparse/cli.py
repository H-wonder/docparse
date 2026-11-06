"""Command line interface: parse documents, or diagnose the local stack."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import __version__, backend, layout, writers
from .config import BASE_URL, MODEL, ParseOptions
from .pipeline import Parser


def cmd_doctor(_: argparse.Namespace) -> int:
    print(f"docparse {__version__}")
    print(f"backend : {BASE_URL}")
    print(f"model   : {MODEL}")
    print(f"ollama  : {'reachable' if backend.ping() else 'NOT reachable'}")
    try:
        import urllib.request
        with urllib.request.urlopen(BASE_URL + "/models", timeout=5) as resp:
            models = json.loads(resp.read()).get("data", [])
        names = [m.get("id", "?") for m in models]
        print(f"models  : {', '.join(names) if names else '(none pulled)'}")
    except Exception as exc:
        print(f"models  : <error: {exc}>")
    print(f"layout  : {'PP-DocLayoutV3 available' if layout.available() else 'not installed (page mode only)'}")
    for dep in ("fitz", "pypdfium2", "PIL"):
        try:
            __import__(dep)
            print(f"{dep:<8}: ok")
        except Exception:
            print(f"{dep:<8}: MISSING")
    return 0


def cmd_parse(args: argparse.Namespace) -> int:
    if not backend.ping():
        print("error: inference backend is not reachable; run scripts/start_ollama.sh", file=sys.stderr)
        return 2
    fields = [f.strip() for f in (args.fields or "").split(",") if f.strip()]
    opts = ParseOptions(
        dpi=args.dpi,
        max_pages=args.max_pages,
        mode=args.mode,
        max_tokens=args.max_tokens,
        retry_repetition=not args.no_retry,
        kie_fields=fields or None,
    )
    parser = Parser(opts)
    outdir = Path(args.outdir)
    failures = 0
    for raw in args.inputs:
        path = Path(raw)
        if not path.exists():
            print(f"skip: {path} (not found)", file=sys.stderr)
            failures += 1
            continue
        t0 = time.perf_counter()
        try:
            doc = parser.parse(path)
        except Exception as exc:
            print(f"FAIL {path}: {exc}", file=sys.stderr)
            failures += 1
            continue
        elapsed = time.perf_counter() - t0
        paths = writers.write(doc, outdir, write_json=not args.markdown_only)
        modes = ",".join(sorted({p.mode for p in doc.pages if p.mode})) or "-"
        flag = "REVIEW" if doc.needs_review else "ok"
        print(f"{flag:6} {path.name:<32} {len(doc.pages):>3}p  {elapsed:5.1f}s  {modes:<12} -> {paths['markdown']}")
        for w in doc.warnings[:5]:
            if w.get("severity") in {"high", "medium"}:
                print(f"        - [{w['severity']}] {w['code']}: {w.get('message','')}")
        if len(doc.warnings) > 5:
            print(f"        ... {len(doc.warnings) - 5} more warnings (see JSON)")
    return 1 if failures else 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .evaluate import evaluate, write_report
    from .pipeline import Parser

    exts = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
    inputs: list[Path] = []
    for raw in args.inputs:
        p = Path(raw)
        if p.is_dir():
            inputs.extend(sorted(x for x in p.iterdir() if x.suffix.lower() in exts))
        elif p.suffix.lower() in exts:
            inputs.append(p)
    if not inputs:
        print("no inputs found", file=sys.stderr)
        return 2

    parser = Parser(ParseOptions(mode=args.mode, dpi=args.dpi))
    result = evaluate(inputs, Path(args.refs), parser)
    write_report(result, Path(args.out))

    print(f"{'input':<36} {'sim':>6} {'cer':>6} {'len':>5}  mode")
    for row in result["rows"]:
        if row["status"] != "ok":
            print(f"{Path(row['input']).name:<36} {row['status']}")
            continue
        print(f"{Path(row['input']).name:<36} {row['similarity']:>6.3f} {row['cer']:>6.3f} "
              f"{row['length_ratio']:>5.2f}  {row.get('mode','')}")
    print("\nsummary:", json.dumps(result["summary"], ensure_ascii=False))
    print(f"report : {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="docparse", description="Local multimodal document parsing (Xiaomi-OCR-0)")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("doctor", help="check backend, model and layout runtime")
    d.set_defaults(fn=cmd_doctor)

    q = sub.add_parser("parse", help="parse PDFs / images to Markdown + IR JSON")
    q.add_argument("inputs", nargs="+", help="files or shell globs already expanded")
    q.add_argument("-o", "--outdir", default="output", help="output directory (default: output)")
    q.add_argument("--mode", choices=["auto", "page", "region"], default="auto")
    q.add_argument("--dpi", type=int, default=200, help="render DPI for scanned pages (default 200)")
    q.add_argument("--max-pages", type=int, default=None)
    q.add_argument("--max-tokens", type=int, default=8192)
    q.add_argument("--no-retry", action="store_true", help="disable automatic repetition retry")
    q.add_argument("--fields", default="",
                   help="comma-separated KIE fields, e.g. '发票号码,开票日期,价税合计'")
    q.add_argument("--markdown-only", action="store_true")
    q.set_defaults(fn=cmd_parse)

    e = sub.add_parser("eval", help="score parse output against reference Markdown")
    e.add_argument("inputs", nargs="+", help="image/PDF files or a directory")
    e.add_argument("--refs", required=True, help="directory containing <stem>.md references")
    e.add_argument("--mode", choices=["auto", "page", "region"], default="auto")
    e.add_argument("--dpi", type=int, default=200)
    e.add_argument("-o", "--out", default="eval_report.json")
    e.set_defaults(fn=cmd_eval)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
