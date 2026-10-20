### Robustness: performance under different conditions

| Condition | Precision | Recall | mAP@0.5 | Change vs clean | Flagged by quality check |
|---|---|---|---|---|---|
| clean | 0.604 | 0.511 | 0.561 | +0% | 5% |
| low_light | 0.583 | 0.508 | 0.530 | -5% | 100% |
| motion_blur | 0.441 | 0.028 | 0.016 | -97% | 52% |
| rain | 0.053 | 0.018 | 0.002 | -100% | 24% |
| glare | 0.536 | 0.480 | 0.483 | -14% | 96% |
| fog | 0.468 | 0.410 | 0.400 | -29% | 79% |
| jpeg_compression | 0.380 | 0.185 | 0.169 | -70% | 8% |
