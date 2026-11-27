# Benchmark results

The released configuration is the unbudgeted pooled 9×9 PAGE method. Counts are successful episodes over the stated evaluation set. Official vanilla entries are the published clean-trained π0.5 values used for orientation.

## RoboTwin 2.0 with π0.5

| Task | Official vanilla C2C | Official vanilla C2R | PAGE C2C | PAGE C2R |
|---|---:|---:|---:|---:|
| `lift_pot` | 98/100 | 35/100 | 97/100 | 26/50 |
| `beat_block_hammer` | 93/100 | 23/100 | 92/100 | 19/50 |
| `rotate_qrcode` | 90/100 | 21/100 | 84/100 | 29/50 |
| `handover_mic` | 98/100 | 6/100 | 99/100 | 21/50 |
| `stack_blocks_two` | 91/100 | 43/100 | 89/100 | 32/50 |
| `move_can_pot` | 60/100 | 10/100 | 60/100 | 18/50 |
| `place_object_basket` | 70/100 | 29/100 | 70/100 | 31/50 |
| **Mean success rate** | **85.7%** | **23.9%** | **84.4%** | **50.3%** |

The official benchmark entries use 100 trials per track. PAGE C2R uses 50 trials per task in the released evaluation.

## VLA-Arena

| Frozen backbone | Official vanilla L0 | Official vanilla L1 | PAGE L0 | PAGE L1 |
|---|---:|---:|---:|---:|
| π0.5 | 88% | 6% | 132/150 (88.0%) | 51/150 (34.0%) |
| OpenVLA-OFT | 96% | 14% | 144/150 (96.0%) | 51/150 (34.0%) |
| **Mean success rate** | **92.0%** | **10.0%** | **92.0%** | **34.0%** |

## AgileX Cobot Magic physical evaluation

Both methods completed 25/25 clean trials. Under visual interference, frozen vanilla achieved 14/25 and PAGE achieved 20/25.
