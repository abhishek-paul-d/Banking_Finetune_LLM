| metric | qwen3_5_4b | gemma4_e4b |
|---|---|---|
| zero-shot macro-F1 (770 ex.) | 0.5630 | 0.6215 |
| zero-shot JSON valid | 1.0000 | 1.0000 |
| fine-tuned intent accuracy | 0.8981 | 0.8964 |
| fine-tuned macro-F1 | 0.6437 | 0.6933 |
| fine-tuned JSON valid | 1.0000 | 1.0000 |
| urgency accuracy | 0.9721 | 0.9682 |
| needs_human accuracy | 0.9854 | 0.9838 |
| train time (s) | 3906.4 | 2340.7 |
| train peak VRAM (GB) | 5.7600 | 12.42 |
| final val loss | 0.0156 | 0.0157 |
| eval gen tokens/s (bs32) | 68.2 | 58 |
| eval peak VRAM (GB) | 4.6000 | 9.9800 |

**TIE on quality (diff -0.0496, 95% CI [-0.0842, +0.0239]). Tie-break by speed -> qwen3_5_4b. Re-run with a second seed before committing.**
