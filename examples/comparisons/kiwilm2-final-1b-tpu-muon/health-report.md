# Matched block-health distributions

100 FP32 MPS batches per checkpoint, batch 2, context 512; 50 batches each from data seeds 141 and 142.

| Model | Block | Mixer | MLP | Amplification median | p90 | Maximum | Batches above 1.5 | Update/residual median |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Dense Muon 0.01 500M | 0 | gqa | swiglu | 0.654 | 0.657 | 0.658 | 0 | 0.501 |
| Dense Muon 0.01 500M | 1 | conv | swiglu | 0.758 | 0.763 | 0.769 | 0 | 0.382 |
| Dense Muon 0.01 500M | 2 | conv | swiglu | 0.892 | 0.896 | 0.902 | 0 | 0.318 |
| Dense Muon 0.01 500M | 3 | gqa | swiglu | 0.928 | 0.933 | 0.937 | 0 | 0.290 |
| Dense Muon 0.01 500M | 4 | conv | swiglu | 1.040 | 1.044 | 1.049 | 0 | 0.287 |
| Dense Muon 0.01 500M | 5 | conv | swiglu | 1.105 | 1.161 | 1.189 | 0 | 0.368 |
| Dense Muon 0.01 500M | 6 | gqa | swiglu | 1.015 | 1.029 | 1.038 | 0 | 0.340 |
| Dense Muon 0.01 500M | 7 | conv | swiglu | 1.113 | 1.122 | 1.137 | 0 | 0.340 |
| Dense Muon 0.01 500M | 8 | conv | swiglu | 1.182 | 1.198 | 1.214 | 0 | 0.419 |
| Dense Muon 0.01 500M | 9 | gqa | swiglu | 1.291 | 1.349 | 1.379 | 0 | 0.737 |
| Dense Muon 0.01 TPU 1B | 0 | gqa | swiglu | 0.617 | 0.621 | 0.625 | 0 | 0.543 |
| Dense Muon 0.01 TPU 1B | 1 | conv | swiglu | 0.735 | 0.743 | 0.749 | 0 | 0.427 |
| Dense Muon 0.01 TPU 1B | 2 | conv | swiglu | 0.890 | 0.895 | 0.901 | 0 | 0.350 |
| Dense Muon 0.01 TPU 1B | 3 | gqa | swiglu | 0.908 | 0.916 | 0.927 | 0 | 0.309 |
| Dense Muon 0.01 TPU 1B | 4 | conv | swiglu | 1.022 | 1.026 | 1.031 | 0 | 0.293 |
| Dense Muon 0.01 TPU 1B | 5 | conv | swiglu | 1.342 | 1.440 | 1.514 | 1 | 0.807 |
| Dense Muon 0.01 TPU 1B | 6 | gqa | swiglu | 1.017 | 1.021 | 1.025 | 0 | 0.266 |
| Dense Muon 0.01 TPU 1B | 7 | conv | swiglu | 1.076 | 1.086 | 1.095 | 0 | 0.285 |
| Dense Muon 0.01 TPU 1B | 8 | conv | swiglu | 1.136 | 1.152 | 1.164 | 0 | 0.365 |
| Dense Muon 0.01 TPU 1B | 9 | gqa | swiglu | 1.177 | 1.244 | 1.282 | 0 | 0.762 |
