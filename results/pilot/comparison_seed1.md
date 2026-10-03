| metric | qwen3_5_4b_seed1 | gemma4_e4b_seed1 |
|---|---|---|
| zero-shot macro-F1 (770 ex.) | nan | nan |
| zero-shot intent accuracy | nan | nan |
| zero-shot JSON valid | nan | nan |
| fine-tuned intent accuracy | 0.9016 | 0.9039 |
| fine-tuned macro-F1 (77 real labels) | 0.9045 | 0.9087 |
| invented-label rate | 0.0075 | 0.0110 |
| fine-tuned JSON valid | 1.0000 | 1.0000 |
| urgency accuracy | 0.9714 | 0.9708 |
| needs_human accuracy | 0.9828 | 0.9825 |
| train time (s) | 4122.7 | 2202.6 |
| train peak VRAM (GB) | 5.7600 | 12.42 |
| final val loss | 0.0166 | 0.0163 |
| eval gen tokens/s (bs16) | 70.8 | 59.9 |
| eval peak VRAM (GB) | 4.6000 | 9.9800 |

**TIE on quality (diff -0.0042, 95% CI [-0.0122, +0.0037]). Tie-break by speed -> qwen3_5_4b_seed1. Re-run with a second seed before committing.**
