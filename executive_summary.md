**Project**: Classification of Ancient Coins  
**Members**: Steven Alex Bradt, Yael Eisenberg & Patrick Millican  
**Instructors**: Lindsay Warrenburg & Marcos Ortiz   
**Program**: Erdős Institute Deep Learning Boot Camp, Summer 2026

**Introduction:**  
The modern traveler, archaeologist, or numismatist has several resources available to identify unfamiliar modern coins, such as [https://coinoscope.com/](https://coinoscope.com/)[^1]. The user simply photographs a coin, uploads it to the app, and receives information on the coin's type, mint years, issuing country, value, and more. Because modern coins are relatively uniform and geometrically crisp, the standard CNN methods used by Coinoscope and similar tools achieve very high success rates. Ancient coins, however, are far from uniform: they were hand-struck, with irregular shapes and sizes, and thousands of years of wear and corrosion have further altered their texture and color. As a result, identifying and classifying ancient coins demands more sophisticated computer vision techniques.

**Datasets:**  
The model was trained, tested, and validated on visual and textual coin data scraped from the Corpus Nummorum database of Greco-Roman coins[^2] (GRECO-ROMAN); and then further tested on Seleucid-dynasty Persian[^3] (SELEUCID), Argead-dynasty Macedonian[^4] (PELLA), and Indo-Greek Bactrian[^5] (BIGR) coin datasets, all three of which come from the American Numismatics Society coin database.

**Approach:**  
First, we built standard scrapers to scrape coin data from the Corpus Nummorum and Numismatics websites, which were designed by academics to permit easy scraping and downloading for research purposes. We then followed the base model architecture methods from CoinNet[^6] to train (with validation) and test a ViT-Base Vision Transformer (vit\_base\_patch16\_224.augreg\_in21k) to recognize cross-culturally common visual motifs on ancient coins from the Greco-Roman period; namely, **eagle**, **throne**, **snake**, **bull**, **horse**, **star**, and **head**. Lastly, we tested the model on coins from temporally and spatially nearby civilizations to determine how extensible our model’s ability to recognize those motifs is.

**Results:**  
The initial training/validation/testing run on the GRECO-ROMAN coins achieved a mean average precision (mAP) of 0.824, with categorical results ranging from 0.730 for **snake** to 0.911 for **head**. Further testing on PELLA coins yielded a mAP of 0.386 (0.010 for **snake** \- 1.000 for **head**), BIGR 0.384 (0.059 for **eagle** \- 1.000 for **head**), and SELEUCID 0.396 (0.011 for **star** \- 0.953 for **head**). This indicates a lack of success in extending the model from GRECO-ROMAN to other types of coins, whether because of idiosyncrasies in categorization or cultural differences in portrayal of the motifs. **Head**, **horse**, and **throne** tended to have greater cross-cultural recognizability than other motifs.

**Further research:**  
Our model can be tested on other coin datasets (particularly others from the Numismatics database) and re-trained/tested on a broader or different set of cross-culturally recognizable motifs. One especially promising direction would be to choose a handful of coin datasets, select a list of motifs with sufficient presence in all the datasets, train a model on each dataset, test each model on all the other datasets, and form a sort of correlation matrix between the models with a mAP score that acts as a proxy for cross-cultural similarity in rendering visual motifs when minting coins.

[^1]:  [https://coinoscope.com/](https://coinoscope.com/)

[^2]:  [https://www.corpus-nummorum.eu/en](https://www.corpus-nummorum.eu/en)

[^3]:  [https://numismatics.org/sco/results](https://numismatics.org/sco/results?q=&start=0)

[^4]:  [https://numismatics.org/pella/results](https://numismatics.org/pella/results)

[^5]:  [https://numismatics.org/bigr/results](https://numismatics.org/bigr/results)

[^6]:  [https://github.com/saeed-anwar/CoinNet](https://github.com/saeed-anwar/CoinNet)
