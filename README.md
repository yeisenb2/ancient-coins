# ancient-coins
Our project utilizes deep learning to analyze and classify ancient coins.
Our approach involves fine-tuning of an instance of `timm` 21k ImageNet, a PyTorch computer vision model, on a dataset of Greco-Roman coins and their metadata from the _Corpus Nummorum_ database; and transfer learning on three other datasets of coins from temporally and geographically nearby civilizations.

The results of our testing are as follows:
| Dataset | Mean average precision (mAP) | HEAD | HORSE | THRONE | STAR | BULL | EAGLE | SNAKE |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| CORPUS NUMMORUM | 0.824 | 0.911 | 0.886 | 0.831 | 0.829 | 0.828 | 0.756 | 0.730 |
| PELLA | 0.505 | 1.000 |   |   |   |   |   | 0.010 | 
| BIGR | 0.384 | 0.592 | 1.000 | 0.431 | 0.110 | 0.112 | 0.059 |   |
| SELEUCID | 0.386 | 0.953 | 0.112 | 0.821 | 0.011 | 0.035 |   |   |

The Corpus Nummorum Greco-Roman coins can be found at [Zenodo](https://zenodo.org/records/10033993).
Other coin datasets are scraped by model_testing.ipynb from the following American Numismatics Society collections:
* [PELLA](https://numismatics.org/pella/) (Argead Dynasty Macedonia)
* [BIGR](https://numismatics.org/bigr/) (Indo-Greek Bactria)
* [SCO](https://numismatics.org/sco/) (Seleucid Dynasty Persia)
