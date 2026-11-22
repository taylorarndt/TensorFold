#!/usr/bin/env python3
"""Build isolated flag-zero/default CUDA runners, dump raw contract outputs, and compare them."""

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
    "block_attention.cu", "common/mla.cuh", "deepseek_v41/attn_cuda.cu",
    "deepseek_v41/attn_split.cuh", "drama3_attn.cuh", "drama3_codec.cu",
    "flashnext_attn.cu", "glm_dflash.cu", "glm_glue_attn.cuh", "h3_attention.cuh",
    "hy4_flash/attention_sink.cu", "hy4_flash/mla_attention.cu", "idol_attention.cuh",
    "idol_decoder_attention.cuh", "inkling2/attention.cu", "inkling2/prompt_attention.cu",
    "minimax_m3_attn.cu", "moss_image_encoder.cu", "muse_spark/attention.cu",
    "muse_spark/dflash.cu", "nemotron_attention.cu", "paged_attn_segs.cuh",
    "prefill_attention.cu", "prefill_attention_rows.cuh", "qwen35.cu",
    "qwen35_attn.cuh", "qwen35_attn_tc.cuh", "qwen35_draft.cu",
    "qwen35_draft_frame.cuh", "qwen35_kv8.cu", "qwen36_mtp.cu", "qwen_vision.cu",
    "tree_attention.cu", "window_attention.cu", "window_rows.cuh", "window_split.cuh",
)


class ABError(RuntimeError):
    pass


def run(command, cwd, env=None):
    result = subprocess.run(
        command, cwd=cwd, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, check=False,
    )
    return result.returncode, result.stdout


def git(repo, *args):
    code, text = run(("git", *args), repo)
    if code:
        raise ABError(f"git {' '.join(args)} failed:\n{text}")
    return text


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources(repo, base):
    listed = tuple(git(repo, "diff", "--name-only", base, "--", "zig/kernels/cuda").splitlines())
    expected = tuple("zig/kernels/cuda/" + name for name in CHANGED)
    if listed != expected:
        raise ABError(f"changed CUDA inventory differs from the controlled 36 files:\n{listed!r}")
    hashes = {}
    control = "TF_CUDA_SOFTMAX_IDENTITY_GUARD"
    for name in CHANGED:
        path = repo / "zig/kernels/cuda" / name
        data = path.read_bytes()
        text = data.decode()
        required = (f"#ifndef {control}\n#define {control} 1\n#endif\n", f"{control} && ")
        if any(token not in text for token in required):
            raise ABError(f"missing default-on compile-time identity control: {path}")
        hashes[name] = hashlib.sha256(data).hexdigest()
    return hashes

def discover_deepseek_replays(root):
    """Select actual recorded attention calls; never invent replacement shapes."""
    root = Path(root).resolve()
    candidates = [root] if (root / "manifest.json").is_file() else sorted(
        path for path in root.iterdir() if path.is_dir() and (path / "manifest.json").is_file()
    )
    cases = []
    for path in candidates:
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest.get("params", {}).get("kind") != "replay":
            continue
        call = manifest.get("call", {})
        if (call.get("ext") == "tf_dsv41_attn_cuda_v1" and call.get("func") == "attn"
                and len(call.get("args", [])) == 23):
            cases.append(str(path))
    if not cases:
        raise ABError(f"no recorded 23-argument DeepSeek attention replay under {root}")
    return cases



def make_overlay(repo, root):
    """A complete symlinked view of the candidate sources for flag-zero compilation."""
    production = repo / "zig/kernels/cuda"
    if root.exists():
        shutil.rmtree(root)
    for source in production.rglob("*"):
        relative = source.relative_to(production)
        target = root / relative
        if source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(source, target)

def make_baseline_nvcc(path, nvcc, overlay, production):
    text = f"""#!/usr/bin/env python3
import os
import sys
NVCC = {str(nvcc)!r}
OVERLAY = {str(overlay)!r}
PRODUCTION = {str(production)!r}
args = []
for arg in sys.argv[1:]:
    normalized = os.path.normpath(arg)
    relative = os.path.relpath(os.path.abspath(normalized), PRODUCTION)
    replaced = arg
    if relative != '..' and not relative.startswith('..' + os.sep):
        target = os.path.join(OVERLAY, relative)
        if os.path.exists(target):
            replaced = target
    args.append(replaced)
# The overlay changes include resolution only; the same current sources select their original arithmetic here.
os.execv(NVCC, [NVCC, '-DTF_CUDA_SOFTMAX_IDENTITY_GUARD=0', '-I' + OVERLAY] + args)
"""
    path.write_text(text)
    path.chmod(0o755)


def build_and_run(repo, args, base, side, nvcc):
    out = Path(args.out).resolve()
    root = out / side
    if root.exists():
        shutil.rmtree(root)
    raw = root / "raw"
    raw.mkdir(parents=True)
    prefix = root / "prefix"
    cache = root / "cache"
    tool = nvcc
    overlay = out / "baseline-overlay"
    if side == "baseline":
        make_overlay(repo, overlay)
        tool = out / "nvcc-baseline"
        make_baseline_nvcc(tool, nvcc, overlay, repo / "zig/kernels/cuda")
    command = [
        args.zig, "build", "cuda-contract-ab-runner",
        "--prefix", str(prefix), "--cache-dir", str(cache),
        "-Dnvcc=" + str(tool), "-Dsm=" + args.sm,
        "-Dcuda-test-dump=true", "-Doptimize=ReleaseFast",
    ]
    env = os.environ.copy()
    env.update(LC_ALL="C", TZ="UTC", SOURCE_DATE_EPOCH="0", PYTHONDONTWRITEBYTECODE="1")
    code, log = run(command, repo, env)
    (root / "build.log").write_text(log)
    if code:
        raise ABError(f"{side} build failed (see {root / 'build.log'}):\n{log[-8000:]}")
    runner = prefix / "bin/tf-cuda-test"
    if not runner.is_file():
        raise ABError(f"{side} build did not install {runner}")
    deepseek_command = None
    if args.deepseek_cases:
        deepseek_command = [
            args.zig, "build", "dsv41-test",
            "--prefix", str(prefix), "--cache-dir", str(cache),
            "-Ddsv41-nvcc=" + str(tool),
        ]
        code, deepseek_log = run(deepseek_command, repo, env)
        with (root / "build.log").open("a") as log_file:
            log_file.write("\n--- DeepSeek replay runner ---\n" + deepseek_log)
        if code:
            raise ABError(f"{side} DeepSeek build failed (see {root / 'build.log'}):\n{deepseek_log[-8000:]}")
        deepseek_runner = prefix / "bin/tf-dsv41-test"
        if not deepseek_runner.is_file():
            raise ABError(f"{side} build did not install {deepseek_runner}")
    else:
        deepseek_runner = None
    provenance = {
        "role": side,
        "base": base,
        "sm": args.sm,
        "command": command,
        "runner": str(runner),
        "runner_sha256": sha(runner),
        "nvcc": str(nvcc),
        "nvcc_wrapper": str(tool) if side == "baseline" else None,
        "overlay": str(overlay) if side == "baseline" else None,
        "baseline_sha256": args.baseline_hashes if side == "baseline" else None,
        "deepseek_cases": args.deepseek_cases,
        "deepseek_fixture_sha256": args.deepseek_fixture_hashes,
        "deepseek_command": deepseek_command,
        "deepseek_runner_sha256": sha(deepseek_runner) if deepseek_runner else None,
    }
    (root / "build.json").write_text(json.dumps(provenance, indent=2) + "\n")

    env["TF_CUDA_IDENTITY_DUMP"] = str(raw)
    code, log = run((str(runner), "kernel-contract"), repo, env)
    (root / "run.log").write_text(log)
    controls = len(re.findall(r"(?m)^CONTROL .*: red at ", log))
    missed = len(re.findall(r"(?m)^CONTROL MISSED ", log))
    if code or not controls or missed:
        raise ABError(
            f"{side} contract failed: exit={code}, red controls={controls}, missed={missed}; "
            f"no incomplete A/B comparison was made (see {root / 'run.log'}):\n{log[-8000:]}"
        )
    for index, case in enumerate(args.deepseek_cases):
        code, log = run((str(deepseek_runner), "replay", case), repo, env)
        path = root / f"deepseek-{index:04d}.log"
        path.write_text(log)
        case_controls = len(re.findall(r"(?m)^CONTROL .*: red at ", log))
        case_missed = len(re.findall(r"(?m)^CONTROL MISSED ", log))
        controls += case_controls
        missed += case_missed
        if (code or "BITEXACT replay tf_dsv41_attn_cuda_v1.attn" not in log
                or "2 R1 variants" not in log or "IDENTITY " not in log or case_missed or case_controls != 2):
            raise ABError(
                f"{side} DeepSeek replay failed for {case}: exit={code}, red controls={case_controls}, missed={case_missed}; "
                f"no incomplete A/B comparison was made (see {path}):\n{log[-8000:]}"
            )
    return runner, controls

def run_logs(root):
    return [root / "run.log"] + sorted(root.glob("deepseek-*.log"))


def tree_hash(root):
    digest = hashlib.sha256()
    root = Path(root)
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()



def receipts(root):
    raw = root / "raw"
    result = {}
    for log_path in run_logs(root):
        for line in log_path.read_text().splitlines():
            if not line.startswith("IDENTITY "):
                continue
            record = json.loads(line[len("IDENTITY "):])
            if record.get("version") != 1:
                raise ABError(f"unknown receipt version in {log_path}")
            key = record["key"]
            if key in result:
                raise ABError(f"duplicate case/shape/control key {key!r} in {side_name(root)}")
            relative = Path(record["file"])
            if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 1:
                raise ABError(f"unsafe output path {relative} in {key}")
            payload = raw / relative
            data = payload.read_bytes()
            if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
                raise ABError(f"payload length/hash mismatch for {side_name(root)} {key}")
            result[key] = (record, data)
    if not result:
        raise ABError(f"no raw IDENTITY receipts in {root}")
    return result


def side_name(root):
    return root.name


def compare(out, base, sm, baseline_hashes):
    sides = {}
    provenance = {}
    controls = {}
    for side in ("baseline", "candidate"):
        root = out / side
        info = json.loads((root / "build.json").read_text())
        if info["role"] != side or info["base"] != base or info["sm"] != sm:
            raise ABError(f"wrong {side} build provenance in {root / 'build.json'}")
        if side == "baseline" and (not info["nvcc_wrapper"] or not info["overlay"]):
            raise ABError("baseline was not built through the flag-zero nvcc overlay")
        if side == "baseline" and info.get("baseline_sha256") != baseline_hashes:
            raise ABError("baseline build does not match the verified current-source hashes")
        if side == "candidate" and info["nvcc_wrapper"] is not None:
            raise ABError("candidate unexpectedly used the baseline wrapper")
        expected_deepseek = len(info.get("deepseek_cases", []))
        if len(list(root.glob("deepseek-*.log"))) != expected_deepseek:
            raise ABError(f"{side} DeepSeek replay log count does not match its build provenance")
        for path in root.glob("deepseek-*.log"):
            replay_log = path.read_text()
            if len(re.findall(r"(?m)^CONTROL .*: red at ", replay_log)) != 2:
                raise ABError(f"{side} DeepSeek replay does not have exactly two red split controls: {path}")
        log = "\n".join(path.read_text() for path in run_logs(root))
        controls[side] = len(re.findall(r"(?m)^CONTROL .*: red at ", log))
        missed = len(re.findall(r"(?m)^CONTROL MISSED ", log))
        if not controls[side] or missed:
            raise ABError(f"{side} planted controls are not all red: red={controls[side]}, missed={missed}")
        provenance[side] = info
        sides[side] = receipts(root)
    if (provenance["baseline"].get("deepseek_cases", []) != provenance["candidate"].get("deepseek_cases", [])
            or provenance["baseline"].get("deepseek_fixture_sha256", {}) != provenance["candidate"].get("deepseek_fixture_sha256", {})):
        raise ABError("baseline and candidate DeepSeek fixture provenance differs")
    if provenance["baseline"]["runner_sha256"] == provenance["candidate"]["runner_sha256"]:
        raise ABError("baseline and candidate runner bytes are identical; refusing a possible two-candidate comparison")
    deepseek_count = len(provenance["baseline"].get("deepseek_cases", []))
    if deepseek_count and provenance["baseline"].get("deepseek_runner_sha256") == provenance["candidate"].get("deepseek_runner_sha256"):
        raise ABError("baseline and candidate DeepSeek runner bytes are identical; refusing a possible two-candidate comparison")
    old, new = sides["baseline"], sides["candidate"]
    common = old.keys() & new.keys()
    missing = sorted(old.keys() - new.keys())
    extra = sorted(new.keys() - old.keys())
    mismatches = []
    metadata = []
    for key in sorted(common):
        before, old_bytes = old[key]
        after, new_bytes = new[key]
        if before["meta"] != after["meta"]:
            metadata.append(key)
            continue
        if old_bytes != new_bytes:
            first = next((i for i, (a, b) in enumerate(zip(old_bytes, new_bytes)) if a != b), min(len(old_bytes), len(new_bytes)))
            mismatches.append({
                "key": key, "kernel": before["meta"]["kernel"],
                "served": before["meta"]["served"], "fill": before["meta"]["fill"],
                "first_different_byte": first, "baseline_bytes": len(old_bytes),
                "candidate_bytes": len(new_bytes), "baseline_sha256": before["sha256"],
                "candidate_sha256": after["sha256"],
            })
    input_cases = {name: sum(1 for record, _ in old.values() if record["meta"]["fill"] == name)
                   for name in ("random", "nonfinite", "stress", "identity", "recorded")}
    clean = not missing and not extra and not mismatches and not metadata
    report = {
        "version": 1, "base": base, "sm": sm, "ok": clean,
        "cases": len(old),
        "input_cases": input_cases, "red_controls": controls, "missing_in_candidate": missing,
        "equal_raw_cases": len(common) - len(mismatches) - len(metadata),
        "missing_in_baseline": extra, "metadata_mismatches": metadata,
        "raw_byte_mismatches": mismatches,
        "runner_sha256": {side: provenance[side]["runner_sha256"] for side in provenance},
        "deepseek_replays": len(provenance["baseline"].get("deepseek_cases", [])),
        "deepseek_runner_sha256": {side: provenance[side].get("deepseek_runner_sha256") for side in provenance},
        "raw_directories": {side: str(out / side / "raw") for side in provenance},
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in (
        "ok", "cases", "input_cases", "equal_raw_cases", "red_controls", "deepseek_replays", "raw_byte_mismatches",
    )}, indent=2))
    print(f"report: {out / 'report.json'}")
    if missing or extra or metadata:
        print(f"missing={len(missing)} extra={len(extra)} metadata={len(metadata)}", file=sys.stderr)
    return clean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nvcc", required=True, help="real nvcc executable (the baseline wraps it)")
    parser.add_argument("--sm", required=True, choices=("120", "121"), help="120=RTX 5090, 121=GB10")
    parser.add_argument("--out", required=True, help="separate build, log, raw-byte, and report directory")
    parser.add_argument("--zig", default="zig")
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--compare-only", action="store_true", help="compare two completed runs; never builds")
    parser.add_argument("--deepseek-replay-root", help="recorded attention case or parent of case directories (SM 121 only)")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    args.baseline_hashes = verify_sources(repo, args.base)
    out = Path(args.out).resolve()
    nvcc = Path(shutil.which(args.nvcc) or "").resolve()
    if not nvcc.is_file():
        raise ABError(f"nvcc not found: {args.nvcc}")
    if args.compare_only:
        return 0 if compare(out, args.base, args.sm, args.baseline_hashes) else 1
    args.deepseek_cases = discover_deepseek_replays(args.deepseek_replay_root) if args.deepseek_replay_root else []
    if args.deepseek_cases and args.sm != "121":
        raise ABError("DeepSeek replay requires --sm 121: dsv41_images.zig compiles family images for sm_121 only")
    args.deepseek_fixture_hashes = {case: tree_hash(case) for case in args.deepseek_cases}
    out.mkdir(parents=True, exist_ok=True)
    for side in ("baseline", "candidate"):
        build_and_run(repo, args, args.base, side, nvcc)
    return 0 if compare(out, args.base, args.sm, args.baseline_hashes) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ABError, OSError, KeyError, ValueError, json.JSONDecodeError) as error:
        print(f"identity A/B error: {error}", file=sys.stderr)
        raise SystemExit(2)
