#!/usr/bin/env python3
"""Build macro-off/default-on Metal runners, dump every served case, compare raw bytes."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

BASE = "25586070b8"
CHANGED = (
    "zig/kernels/metal/core/prompt_attention_nax.metal",
    "zig/kernels/metal/core/shared_attention.metal",
    "zig/kernels/metal/deepseek_v41/csa2_attn.metal",
    "zig/kernels/metal/deepseek_v41/vision.metal",
    "zig/kernels/metal/drama3/attn.metal",
    "zig/kernels/metal/drama3/codec.metal",
    "zig/kernels/metal/flashnext/q4_attn_merge_gate_7070840cd3c22e98.metal",
    "zig/kernels/metal/flashnext/q4_attn_parts_f6b4ffe8e2609998.metal",
    "zig/kernels/metal/flint/attention_nax.metal",
    "zig/kernels/metal/flint/glue.metal",
    "zig/kernels/metal/gemma4/glue.metal",
    "zig/kernels/metal/gemma4/prompt_attention.metal",
    "zig/kernels/metal/glm/sparse_attention.metal",
    "zig/kernels/metal/glm_draft_attention.metal",
    "zig/kernels/metal/glm_dsa/mla.metal",
    "zig/kernels/metal/glm_sparse_nax.metal",
    "zig/kernels/metal/h3/h3_attention.metal",
    "zig/kernels/metal/hy4_flash/attention_sink.metal",
    "zig/kernels/metal/hy4_flash/mla_attention.metal",
    "zig/kernels/metal/hy4_flash/mla_nax.metal",
    "zig/kernels/metal/idol_attention.metal",
    "zig/kernels/metal/idol_decoder.metal",
    "zig/kernels/metal/idol_dino.metal",
    "zig/kernels/metal/idol_vae.metal",
    "zig/kernels/metal/kimi/mla.metal",
    "zig/kernels/metal/lfm2/lfm2.metal",
    "zig/kernels/metal/lfm2/lfm2_attn_nax.metal",
    "zig/kernels/metal/lfm3/attention.metal",
    "zig/kernels/metal/llama/glue.metal",
    "zig/kernels/metal/minimax_m3/attention_nax.metal",
    "zig/kernels/metal/minimax_m3/dspark.metal",
    "zig/kernels/metal/minimax_m3/glue.metal",
    "zig/kernels/metal/moss_image_decoder.metal",
    "zig/kernels/metal/moss_image_encoder.metal",
    "zig/kernels/metal/nemotron/attn_merge.metal",
    "zig/kernels/metal/nemotron/attn_partial.metal",
    "zig/kernels/metal/nemotron_tree.metal",
    "zig/kernels/metal/ops/sdpa.metal",
    "zig/kernels/metal/prefill/attention_nax.metal",
    "zig/kernels/metal/prefill/fn_attn.metal",
    "zig/kernels/metal/prefill/qmm6_nax_b.metal",
    "zig/kernels/metal/qwen27/attention_mpp.metal",
    "zig/kernels/metal/qwen27/attention_prompt.metal",
    "zig/kernels/metal/qwen27/attention_prompt_sg.metal",
    "zig/kernels/metal/qwen3_5/attn_merge.metal",
    "zig/kernels/metal/qwen3_5/attn_partial.metal",
    "zig/kernels/metal/qwen3_5/paged.metal",
    "zig/kernels/metal/spark25/glue.metal",
    "zig/kernels/metal/step5/attend_nax.metal",
    "zig/kernels/metal/step5/sparse.metal",
    "zig/kernels/metal/uocr/vision.metal",
    "zig/src/families/flashnext/attn2.zig",
)
DEVICES = {"m3-ultra": "Apple M3 Ultra", "m5-ultra": "Apple M5 Ultra"}
GPU_GAPS = ()


def git(repo, *args):
    done = subprocess.run(("git", *args), cwd=repo, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, check=False)
    if done.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed:\n{done.stdout}")
    return done.stdout


def verify_sources(repo, base):
    paths = ("zig/kernels/metal", "zig/src/families/flashnext/attn2.zig")
    listed = tuple(git(repo, "diff", "--name-only", base, "--", *paths).splitlines())
    if listed != CHANGED:
        raise RuntimeError(f"changed Metal inventory differs from the expected 52 files:\n{listed!r}")
    hashes = {}
    for source in CHANGED:
        raw = (repo / source).read_bytes()
        text = raw.decode()
        if "TF_METAL_SOFTMAX_IDENTITY_GUARD" not in text:
            raise RuntimeError(f"identity compile-time control missing: {source}")
        if source.endswith(".metal"):
            default = "#ifndef TF_METAL_SOFTMAX_IDENTITY_GUARD\n#define TF_METAL_SOFTMAX_IDENTITY_GUARD 1\n#endif\n"
            if default not in text:
                raise RuntimeError(f"identity default-on control missing: {source}")
        hashes[source] = hashlib.sha256(raw).hexdigest()
    return hashes


def make_overlay(repo, root, out):
    """Baseline view of current sources, with shader identity guards overridden off."""
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    out = repo / ".git"
    if out.exists():
        os.symlink(out, root / ".git")
    skip = {".git", ".zig-cache", "zig-cache", "zig-out"}
    excluded = out.resolve()
    for source in repo.rglob("*"):
        if source == excluded or excluded in source.parents:
            continue
        relative = source.relative_to(repo)
        if any(part in skip for part in relative.parts):
            continue
        target = root / relative
        if source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            if str(relative) in CHANGED and source.suffix == ".metal":
                target.write_bytes(b"#define TF_METAL_SOFTMAX_IDENTITY_GUARD 0\n" + source.read_bytes())
            else:
                os.symlink(source, target)


def execute(command, cwd, env=None):
    done = subprocess.run(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, check=False)
    return done.returncode, done.stdout


def build_and_run(repo, overlay, args, side, baseline_hashes):
    out = Path(args.out).resolve()
    root = out / side
    if root.exists():
        shutil.rmtree(root)
    raw = root / "raw"
    raw.mkdir(parents=True)
    prefix, cache = root / "prefix", root / "cache"
    source = repo if side == "candidate" else overlay
    command = [args.zig, "build", "tf-kernel-contract", "tf-rotary-sweep", "--prefix", str(prefix),
               "--cache-dir", str(cache), "-Doptimize=ReleaseFast"]
    env = os.environ.copy()
    env.update(LC_ALL="C", TZ="UTC", SOURCE_DATE_EPOCH="0", PYTHONDONTWRITEBYTECODE="1")
    code, log = execute(command, source, env)
    (root / "build.log").write_text(log)
    if code:
        raise RuntimeError(f"{side} build failed (see {root / 'build.log'}):\n{log[-8000:]}")
    runner = prefix / "bin/tf-kernel-contract"
    rotary = prefix / "bin/tf-rotary-sweep"
    if not runner.is_file() or not rotary.is_file():
        raise RuntimeError(f"{side} build did not install both runners under {prefix / 'bin'}")
    provenance = {
        "role": side, "base": args.base, "device": args.device, "command": command,
        "runner": str(runner), "runner_sha256": hashlib.sha256(runner.read_bytes()).hexdigest(),
        "source_root": str(source), "sha256": baseline_hashes if side == "baseline" else None,
    }
    (root / "build.json").write_text(json.dumps(provenance, indent=2) + "\n")
    receipt = root / "rotary.json"
    code, log = execute((str(rotary), "--out", str(receipt)), source, env)
    (root / "rotary.log").write_text(log)
    if code:
        raise RuntimeError(f"{side} source-matched rotary sweep failed (see {root / 'rotary.log'}):\n{log[-8000:]}")
    run_command = [str(runner), "--dump", str(raw), "--no-limits", "--seed", "0x6b63",
                   "--rotary-accuracy", str(receipt)]
    code, log = execute(run_command, source, env)
    (root / "run.log").write_text(log)
    controls = len(re.findall(r"(?m)^CONTROL .*: red at ", log))
    missed = len(re.findall(r"(?m)^CONTROL MISSED ", log))
    device = DEVICES[args.device]
    if code or not controls or missed or f"tf-kernel-contract on {device}" not in log:
        raise RuntimeError(
            f"{side} contract invalid: exit={code}, red controls={controls}, missed={missed}, "
            f"required device={device}; see {root / 'run.log'}:\n{log[-8000:]}")


def compare(out, base, device, baseline_hashes):
    sides, controls, provenance = {}, {}, {}
    for side in ("baseline", "candidate"):
        root = out / side
        info = json.loads((root / "build.json").read_text())
        if info["role"] != side or info["base"] != base or info["device"] != device:
            raise RuntimeError(f"wrong {side} build provenance")
        if side == "baseline" and info["sha256"] != baseline_hashes:
            raise RuntimeError("baseline is not built from the verified current source hashes")
        if side == "candidate" and info["sha256"] is not None:
            raise RuntimeError("candidate unexpectedly carries baseline source hashes")
        log = (root / "run.log").read_text()
        controls[side] = len(re.findall(r"(?m)^CONTROL .*: red at ", log))
        if not controls[side] or re.search(r"(?m)^CONTROL MISSED ", log):
            raise RuntimeError(f"{side} planted controls are not all red")
        provenance[side] = info
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cuda"))
        from identity_ab import receipts
        sides[side] = receipts(root)
    old, new = sides["baseline"], sides["candidate"]
    missing, extra = sorted(old.keys() - new.keys()), sorted(new.keys() - old.keys())
    mismatches, metadata = [], []
    for key in sorted(old.keys() & new.keys()):
        before, old_bytes = old[key]
        after, new_bytes = new[key]
        if before["meta"] != after["meta"]:
            metadata.append(key)
        elif old_bytes != new_bytes:
            first = next((i for i, pair in enumerate(zip(old_bytes, new_bytes)) if pair[0] != pair[1]),
                         min(len(old_bytes), len(new_bytes)))
            mismatches.append({"key": key, "kernel": before["meta"]["kernel"],
                               "served": before["meta"]["served"], "fill": before["meta"]["fill"],
                               "first_different_byte": first, "baseline_sha256": before["sha256"],
                               "candidate_sha256": after["sha256"]})
    input_cases = {name: sum(record["meta"]["fill"] == name for record, _ in old.values())
                   for name in ("random", "nonfinite", "stress", "identity")}
    clean = not missing and not extra and not mismatches and not metadata
    report = {"version": 1, "base": base, "device": device, "ok": clean, "cases": len(old),
              "equal_raw_cases": len(old) - len(mismatches) - len(metadata), "input_cases": input_cases,
              "red_controls": controls, "missing_in_candidate": missing, "missing_in_baseline": extra,
              "metadata_mismatches": metadata, "raw_byte_mismatches": mismatches,
              "runner_sha256": {side: info["runner_sha256"] for side, info in provenance.items()},
              "raw_directories": {side: str(out / side / "raw") for side in provenance}}
    report["complete_changed_source_gpu_coverage"] = not GPU_GAPS
    report["unregistered_changed_sources"] = list(GPU_GAPS)
    report["coverage_note"] = "All 52 changed Metal sources/transforms have GPU registrations; every kernel and served shape was dumped." if not GPU_GAPS else "All registered kernels/served shapes were dumped; these changed sources have no Metal GPU contract registration."
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("ok", "cases", "input_cases", "equal_raw_cases",
                                                   "red_controls", "raw_byte_mismatches")}, indent=2))
    print(f"report: {out / 'report.json'}")
    return clean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, choices=tuple(DEVICES))
    parser.add_argument("--out", required=True, help="separate builds, logs, raw bytes, and report")
    parser.add_argument("--zig", default="zig")
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--compare-only", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    hashes = verify_sources(repo, args.base)
    out = Path(args.out).resolve()
    if args.compare_only:
        return 0 if compare(out, args.base, args.device, hashes) else 1
    out.mkdir(parents=True, exist_ok=True)
    overlay = out / "baseline-source"
    make_overlay(repo, overlay, out)
    for side in ("baseline", "candidate"):
        build_and_run(repo, overlay, args, side, hashes)
    return 0 if compare(out, args.base, args.device, hashes) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, KeyError, ValueError, json.JSONDecodeError) as error:
        print(f"Metal identity A/B error: {error}", file=sys.stderr)
        raise SystemExit(2)
