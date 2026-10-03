| variant | size (GB) | intent acc | delta vs fp16 | macro-F1 | invented labels | valid JSON | urgency | needs_human | same answer as fp16 |
|---|---|---|---|---|---|---|---|---|---|
| fp16 merged | 9.10 | 0.9373 | +0.0000 | 0.9375 | 0.06% | 100.00% | 0.9831 | 0.9912 | 100.00% |
| GGUF Q8_0 | 4.48 | 0.9383 | +0.0010 | 0.9385 | 0.06% | 100.00% | 0.9834 | 0.9912 | 99.81% |
| GGUF Q4_K_M | 2.71 | 0.9357 | -0.0016 | 0.9361 | 0.13% | 100.00% | 0.9799 | 0.9899 | 98.67% |
| AWQ default | 5.37 | 0.9312 | -0.0062 | 0.9314 | 0.13% | 100.00% | 0.9799 | 0.9906 | 97.86% |
| AWQ mlp_only | 5.80 | 0.9331 | -0.0042 | 0.9333 | 0.13% | 100.00% | 0.9799 | 0.9903 | 98.28% |

Notes
- Sizes: fp16, Q8_0 and Q4_K_M measured in GB (1e9 bytes). AWQ sizes were read from `du -h` (5.0 GiB and 5.4 GiB) and converted, so they are approximate (+-0.05 GB).
- The fp16 folder (9.10 GB) includes the unused vision tower (0.67 GB); GGUF files are text-only.
- AWQ is 4-bit only for the MLP (and, in `default`, the standard attention) layers. The new `linear_attn` layers (2.0 GB), the embeddings (1.3 GB) and the vision tower (0.7 GB) stay in fp16, which is why the AWQ files are larger than Q8_0.
- Scores are on the same 3,080 held-out examples. Differences below ~0.3 points are within noise (paired McNemar tests: Q8_0 p=0.25, Q4_K_M p=0.44, AWQ mlp_only p=0.04, AWQ default p<0.01, all vs fp16).
