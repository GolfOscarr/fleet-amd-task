# Measurement report

| Quantity | Predicted | Measured |
|---|---|---|
| bytes per iteration (read, MiB) | 4709.9 weights + cache; about 4772.0 with activations | - |
| time per iteration, median (us) | 1148.5-1349.4 + 326 t_b + T_serial | - |
| time per iteration, P95 (us) |  | - |
| time per iteration from event timing, median (us) |  | 377.5 |
| time per iteration from host wall clock (us) |  | 913.5 |
| time per iteration from the kernel trace (us) |  | - |
| achieved read bandwidth (TB/s) | 3.66-4.3 over T_bw | - |
| launches per generation | 3 | - |
| L2 hit rate | 16-17% (Fleet's batch-1 figure) | - |
| tokens per second |  | - |
| GFX clock from amd-smi during the run, median / max (MHz) | 2100 max (D2 of round 1) | 2107 / 2108 over 23 samples |
| memory clock from amd-smi, median (MHz) |  | 900 |

## Per-operator time and rate (stream probe, 14.2 MiB per operator)

| Event | Operator | mean us | min us | max us | n | GB/s |
|---|---|---|---|---|---|---|
| 1 | iteration_start | 177.03 | 169.52 | 205.96 | 30 | - |
| 2 | stream_layer | 30.96 | 29.58 | 32.32 | 31 | 482.6 |
| 3 | stream_layer | 19.32 | 17.72 | 21.26 | 31 | 773.3 |
| 4 | stream_layer | 19.35 | 17.56 | 21.52 | 31 | 772.0 |
| 5 | stream_layer | 19.28 | 17.40 | 21.00 | 31 | 775.1 |
| 6 | stream_layer | 19.31 | 17.44 | 21.32 | 31 | 773.7 |
| 7 | stream_layer | 19.23 | 17.40 | 21.40 | 31 | 777.2 |
| 8 | stream_layer | 19.04 | 17.16 | 21.64 | 31 | 784.8 |
| 9 | stream_layer | 19.20 | 16.64 | 21.40 | 31 | 778.4 |
| 10 | stream_layer | 19.35 | 17.20 | 21.48 | 31 | 772.0 |
| 11 | stream_layer | 18.88 | 17.00 | 20.96 | 31 | 791.4 |

stream probe: 775.1 GB/s, the median over the 9 operators after the first
