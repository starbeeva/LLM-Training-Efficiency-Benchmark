| config | forward ms | backward ms | peak GB | tokens/s |
|---|---|---|---|---|
| baseline fp32 with custom layers | 258.22 ± 69.672 | 536.839 ± 139.534 | 9.052 | 2412 |
| baseline fp32 with custom layers | 231.863 ± 148.801 | 413.916 ± 318.343 | 7.282 | 2926 |
| baseline fp32 | 236.869 ± 104.594 | 502.606 ± 135.466 | 8.849 | 2581 |
| fp32 compiled | 227.81 ± 180.633 | 401.812 ± 494.487 | 7.28 | 2996 |
| baseline fp16 | 124.016 ± 4.219 | 285.881 ± 69.731 | 6.981 | 4351 |
| fp16 compiled | 52.289 ± 1.563 | 100.297 ± 45.512 | 5.402 | 9609 |
| baseline bf16 (not native on T4) | 330.767 ± 111.035 | 703.116 ± 305.095 | 6.981 | 1870 |
| baseline fp32 with full checkpointing | 264.591 ± 520.425 | 727.056 ± 25.299 | 3.73 | 1958 |
| fp16 with full checkpointing | 125.08 ± 1090.095 | 385.17 ± 18.787 | 3.527 | 3586 |
| compiled fp16 with full checkpointing | 123.006 ± 10.68 | 394.357 ± 142.458 | 3.527 | 3542 |
| fp16 with SELECTIVE checkpointing | 127.276 ± 1234.028 | 378.294 ± 31.773 | 4.139 | 3615 |
| compiled fp16 with SELECTIVE checkpointing (attn block megatron vibe) | 117.367 ± 2.046 | 360.514 ± 49.355 | 3.92 | 3800 |
