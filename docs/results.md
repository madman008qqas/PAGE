# Benchmark results

The released configuration is the unbudgeted pooled 9×9 PAGE method. Values are success rates over the stated evaluation set. Official vanilla entries are the published clean-trained π0.5 values used for orientation.

## RoboTwin 2.0 with π0.5

| Task | Official vanilla C2C | Official vanilla C2R | PAGE C2C | PAGE C2R |
|---|---:|---:|---:|---:|
| `lift_pot` | 98% | 35% | 97% | 52% |
| `beat_block_hammer` | 93% | 23% | 92% | 38% |
| `rotate_qrcode` | 90% | 21% | 84% | 58% |
| `handover_mic` | 98% | 6% | 99% | 42% |
| `stack_blocks_two` | 91% | 43% | 89% | 64% |
| `move_can_pot` | 60% | 10% | 60% | 36% |
| `place_object_basket` | 70% | 29% | 70% | 62% |
| **Mean success rate** | **85.7%** | **23.9%** | **84.4%** | **50.3%** |

## VLA-Arena

| Frozen backbone | Official vanilla L0 | Official vanilla L1 | PAGE L0 | PAGE L1 |
|---|---:|---:|---:|---:|
| π0.5 | 88% | 6% | 88.0% | 34.0% |
| OpenVLA-OFT | 96% | 14% | 96.0% | 34.0% |
| **Mean success rate** | **92.0%** | **10.0%** | **92.0%** | **34.0%** |

## AgileX Cobot Magic physical evaluation

Both methods completed 25/25 clean trials. Under visual interference, frozen vanilla achieved 14/25 and PAGE achieved 20/25.
