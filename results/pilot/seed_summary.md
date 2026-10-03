| model | seed | intent acc | macro-F1 | invented labels |
|---|---|---|---|---|
| qwen3_5_4b | 0 | 0.8981 | 0.9029 | 1.23% |
| qwen3_5_4b | 1 | 0.9016 | 0.9045 | 0.75% |
| **qwen3_5_4b** | mean (range) | 0.8998 | 0.9037 (0.9029 to 0.9045) | |
| gemma4_e4b | 0 | 0.8964 | 0.9004 | 1.01% |
| gemma4_e4b | 1 | 0.9039 | 0.9087 | 1.10% |
| **gemma4_e4b** | mean (range) | 0.9002 | 0.9045 (0.9004 to 0.9087) | |

Difference in macro-F1, qwen3_5_4b minus gemma4_e4b (paired bootstrap 95% CI):

| seed | diff | 95% CI | verdict for this seed |
|---|---|---|---|
| 0 | +0.0025 | [-0.0055, +0.0115] | tie |
| 1 | -0.0042 | [-0.0122, +0.0037] | tie |
| mean | -0.0008 | | |

qwen3_5_4b: 7.1% of test answers differ between seed 0 and seed 1.

gemma4_e4b: 7.3% of test answers differ between seed 0 and seed 1.

**TIE CONFIRMED across 2 seeds (mean diff -0.0008). The choice rests on speed and memory, not quality.**
