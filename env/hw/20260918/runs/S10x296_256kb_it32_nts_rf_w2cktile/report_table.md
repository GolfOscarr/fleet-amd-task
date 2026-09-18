# Measurement report

| Quantity | Predicted | Measured |
|---|---|---|
| bytes per iteration (read, MiB) | 4709.9 weights + cache; about 4772.0 with activations | - |
| time per iteration, median (us) | 1148.5-1349.4 + 326 t_b + T_serial | - |
| time per iteration, P95 (us) |  | - |
| time per iteration from event timing, median (us) |  | 816.2 |
| time per iteration from host wall clock (us) |  | 1356.6 |
| time per iteration from the kernel trace (us) |  | - |
| achieved read bandwidth (TB/s) | 3.66-4.3 over T_bw | - |
| launches per generation | 3 | - |
| L2 hit rate | 16-17% (Fleet's batch-1 figure) | - |
| tokens per second |  | - |
| GFX clock from amd-smi during the run, median / max (MHz) | 2100 max (D2 of round 1) | 2107 / 2107 over 24 samples |
| memory clock from amd-smi, median (MHz) |  | 900 |

## Per-operator time and rate (stream probe, 74.0 MiB per operator)

| Event | Operator | mean us | min us | max us | n | GB/s |
|---|---|---|---|---|---|---|
| 1 | iteration_start | 202.12 | 193.80 | 217.16 | 30 | - |
| 2 | stream_layer | 72.16 | 68.72 | 73.48 | 31 | 1075.3 |
| 3 | stream_layer | 59.98 | 57.64 | 62.12 | 31 | 1293.7 |
| 4 | stream_layer | 60.38 | 57.40 | 62.00 | 31 | 1285.0 |
| 5 | stream_layer | 60.45 | 58.16 | 62.00 | 31 | 1283.6 |
| 6 | stream_layer | 60.19 | 56.84 | 62.36 | 31 | 1289.1 |
| 7 | stream_layer | 60.65 | 57.76 | 62.48 | 31 | 1279.5 |
| 8 | stream_layer | 60.48 | 57.08 | 62.72 | 31 | 1283.0 |
| 9 | stream_layer | 60.59 | 57.44 | 62.32 | 31 | 1280.7 |
| 10 | stream_layer | 60.63 | 56.56 | 62.44 | 31 | 1279.8 |
| 11 | stream_layer | 60.57 | 57.92 | 62.20 | 31 | 1281.1 |

stream probe: 1283.0 GB/s, the median over the 9 operators after the first
