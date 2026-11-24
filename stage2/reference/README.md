# stage2/reference

This folder holds static inputs that are published beside our own outputs and cannot be regenerated.

| file | what |
|---|---|
| `bbw_factors_original_2004_2021.csv` | The original Bai, Bali and Wen (2019) four-factor series, 2004-08 to 2021-12, as the authors distributed it with the paper (since retracted). It is kept unchanged, so the correction in Dickerson, Mueller and Robotti (2023) can be measured against it. `make_release.py --what bbw` ships it inside `osbap_bbw_factors_<vintage>.zip` and pins its sha256, so a changed file cannot pass as the original. |
