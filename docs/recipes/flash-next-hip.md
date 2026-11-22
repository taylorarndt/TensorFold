# Qwen3.8 Flash Next on HIP

This page is the home for Flash Next on HIP: the native build serving `qwen4_exp` checkpoints on AMD GPUs.
The integrated-GPU platform (gfx1151, Strix Halo) has its own section in the MLX prefill recipe,
[Native HIP on a Strix Halo iGPU](qwen3.8-flash-next.md#native-hip-on-a-strix-halo-igpu), which this page
does not repeat.

Discrete gfx12 cards are covered here by one receipt,
[docs/receipts/gfx1201-flashnext-tp2-f8dacd77](../receipts/gfx1201-flashnext-tp2-f8dacd77/README.md):
two Radeon AI PRO R9700s (gfx1201, 32 GiB each) at `--tp 2`, one host, with
`Qwen3.8-Flash-Next-MLX-oQ2-MTP` at `0a1bf6fa` — decode at `--parallel 4 --context 4096` (121.2 tok/s,
`bench_openai` fibonacci-raw) and cold prefill at `--parallel 1 --context 40960`
(1,191/1,121/1,058 tok/s at 2k/8k/32k tokens), both with `--prompt-cache-gib 0`, every `alone` and
`serial` comparison equal. What follows is that measured path only.

## Two-rank serve path

```bash
# rank 1 (does not serve HTTP)
HIP_VISIBLE_DEVICES=<card> tensorfold-native serve TensorFold/Qwen3.8-Flash-Next-MLX-oQ2-MTP@0a1bf6faa0a2f16a314da889e8e6c80226455dbf --tp 2 --rank 1 --master 127.0.0.1 --prompt-cache-gib 0 --parallel 4 --context 4096
# rank 0 (serves HTTP)
HIP_VISIBLE_DEVICES=<card> tensorfold-native serve TensorFold/Qwen3.8-Flash-Next-MLX-oQ2-MTP@0a1bf6faa0a2f16a314da889e8e6c80226455dbf --tp 2 --rank 0 --master 127.0.0.1 --port 8092 --prompt-cache-gib 0 --parallel 4 --context 4096
```

- Both ranks run the same binary; `--master` names rank 0's host and `--master-port` (default 29551) its
  control port, and the ranks wait for each other.
- `--prompt-cache-gib 0` was used with drafts at `--tp 2`, per the CUDA recipe's qualification terms;
  prompt-cache retention with MTP was not measured.
- HIP enumeration order differs from `rocm-smi` card numbers; verify the target card by occupancy
  (`rocm-smi --showmeminfo vram --csv`) and pin it with `HIP_VISIBLE_DEVICES`, not by trusting an index.
- The weights take 27.3 GiB per rank at `--tp 2`.

Not measured by that receipt: other card counts, `--tp 4`/`--tp 8`, multi-host ranks, other checkpoints
or revisions, other context or parallel windows, fp8 KV, and drafter checkpoints. A host build does not
establish checkpoint fidelity, qualify later `main` commits, or qualify other gfx12 variants.

## Experimental decode tail graphs

`TF_FLASHNEXT_HIP_TAIL_GRAPH=1` captures the post-verify residual copies and greedy argmax, window keeps, and
the serial MTP forwards and draws on one admitted stream and one rank, with the normal graph policy enabled.
Unset or0 keeps the eager tail. Each submission records current arguments and updates its executable graph;
a topology change instantiates a replacement. Draw-result waits remain, and a keep's event protects its
pinned plan source when no later draw has drained it. This changes no kernel or arithmetic.

This is an unqualified opt-in. Host recording tests do not prove GPU token equality or a speedup. Before enabling
it for serving, compare the same checkpoint with baseline, flag0 and flag1: greedy and seeded sampled tokens,
drafts0/3/7, cache resume and cancellation, cold2k/8k prompts, and graph-off controls. Require identical token
hashes before measuring. Record eager/captured calls, copies, full waits and event waits separately; graph
refresh still makes CPU recording calls. On gfx1151, require at least3% paired primary decode gain and no
guard-cell regression above1%. Other HIP devices and multi-stream/multi-rank tail capture are unqualified.
