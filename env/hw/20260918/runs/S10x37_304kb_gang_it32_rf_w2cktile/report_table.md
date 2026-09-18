# Measurement report

| Quantity | Predicted | Measured |
|---|---|---|
| bytes per iteration (read, MiB) | 4709.9 weights + cache; about 4772.0 with activations | - |
| time per iteration, median (us) | 1148.5-1349.4 + 326 t_b + T_serial | - |
| time per iteration, P95 (us) |  | - |
| time per iteration from event timing, median (us) |  | 624.5 |
| time per iteration from host wall clock (us) |  | 1202.9 |
| time per iteration from the kernel trace (us) |  | - |
| achieved read bandwidth (TB/s) | 3.66-4.3 over T_bw | - |
| launches per generation | 3 | - |
| L2 hit rate | 16-17% (Fleet's batch-1 figure) | - |
| tokens per second |  | - |
| GFX clock from amd-smi during the run, median / max (MHz) | 2100 max (D2 of round 1) | 2107 / 2108 over 24 samples |
| memory clock from amd-smi, median (MHz) |  | 900 |

## Per-operator time and rate (stream probe, 87.9 MiB per operator)

| Event | Operator | mean us | min us | max us | n | GB/s |
|---|---|---|---|---|---|---|
| 1 | iteration_start | 197.52 | 9.88 | 228.92 | 30 | - |
| 2 | stream_gang_layer | 55.84 | 54.50 | 57.74 | 31 | 1650.1 |
| 3 | stream_gang_layer | 41.03 | 39.78 | 42.90 | 31 | 2245.6 |
| 4 | stream_gang_layer | 40.95 | 39.80 | 41.92 | 31 | 2250.4 |
| 5 | stream_gang_layer | 40.86 | 39.96 | 42.36 | 31 | 2255.1 |
| 6 | stream_gang_layer | 40.60 | 39.68 | 41.64 | 31 | 2269.6 |
| 7 | stream_gang_layer | 40.96 | 39.92 | 42.40 | 31 | 2249.4 |
| 8 | stream_gang_layer | 40.75 | 39.78 | 41.88 | 31 | 2261.4 |
| 9 | stream_gang_layer | 40.71 | 39.46 | 41.74 | 31 | 2263.7 |
| 10 | stream_gang_layer | 40.82 | 39.60 | 42.20 | 31 | 2257.4 |
| 11 | stream_gang_layer | 40.48 | 39.76 | 41.56 | 31 | 2276.2 |

stream probe: 2257.4 GB/s, the median over the 9 operators after the first
