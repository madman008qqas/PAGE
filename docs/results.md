# Benchmark summary

The numbers below summarize the internal evaluation protocol for the released PAGE configuration. They are included as a compact orientation for readers. Full per-run receipts, machine paths, and private assets are intentionally excluded.

## RoboTwin 2.0 with π0.5

The released pooled 9×9 PAGE configuration reports the following counts. The official vanilla columns are the published clean-trained π0.5 entries used for orientation. Official RoboTwin C2R entries use 100 trials, while this release's PAGE C2R runs use 50 trials per task.

| Task | Official vanilla C2C | Official vanilla C2R | PAGE C2C | PAGE C2R |
|---|---:|---:|---:|---:|
| lift_pot | 98/100 | 35/100 | 97/100 | 26/50 |
| beat_block_hammer | 93/100 | 23/100 | 92/100 | 19/50 |
| rotate_qrcode | 90/100 | 21/100 | 84/100 | 29/50 |
| handover_mic | 98/100 | 6/100 | 99/100 | 21/50 |
| stack_blocks_two | 91/100 | 43/100 | 89/100 | 32/50 |
| stack_blocks_three | 77/100 | 25/100 | 59/100 | 16/50 |
| move_can_pot | 60/100 | 10/100 | 60/100 | 18/50 |
| place_can_basket | 56/100 | 10/100 | 64/100 | 12/50 |
| place_object_basket | 70/100 | 29/100 | 70/100 | 31/50 |
| place_object_scale | 78/100 | 38/100 | 70/100 | 21/50 |
| **Mean success rate** | **81.1%** | **24.0%** | **64.6%** | **39.2%** |

Across the ten tasks, the official averages are 81.1% C2C and 24.0% C2R. PAGE totals are 323/500 C2C (64.6%) and 98/250 C2R (39.2%).

## VLA-Arena

The released 9×9 configuration was evaluated with π0.5 and OpenVLA-OFT under clean (L0) and static-interference (L1) conditions. Aggregate counts are shown below.

| Backbone | Vanilla L0 | Vanilla L1 | PAGE L0 | PAGE L1 |
|---|---:|---:|---:|---:|
| π0.5 | 132/150 | 9/150 | 123/150 | 51/150 |
| OpenVLA-OFT | 144/150 | 21/150 | 144/150 | 51/150 |

On physical AgileX Cobot Magic dual-PiPER hardware, both methods achieved 25/25 clean trials. Under visual interference, frozen vanilla achieved 14/25 and PAGE achieved 20/25.

These summaries identify the method configuration and benchmark conditions. They do not replace the benchmark's official evaluation instructions or provide downloadable model assets.
