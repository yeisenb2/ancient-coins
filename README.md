# ancient-coins
Our project utilizes deep learning to analyze and classify ancient coins.
Our approach involves fine-tuning of an instance of `timm` 21k ImageNet, a PyTorch computer vision model, on a dataset of Greco-Roman coins and their metadata from the _Corpus Nummorum_ database; and transfer learning on three other datasets of coins from temporally and geographically nearby civilizations.

The Corpus Nummorum Greco-Roman coins can be found at [Zenodo](https://zenodo.org/records/10033993).
Other coin datasets are scraped by model_testing.ipynb from the following American Numismatics Society collections:
* [PELLA](https://numismatics.org/pella/) (Argead Dynasty Macedonia)
* [BIGR](https://numismatics.org/bigr/) (Indo-Greek Bactria)
* [SCO](https://numismatics.org/sco/) (Seleucid Dynasty Persia)
