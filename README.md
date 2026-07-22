# BIOSEaRCH

The project Bioinformatics Search, Evaluation and Ranking of Causal Hits (BIOSEaRCH) was created in July 2026 within the Big Challenges programm run by [Educational Centre Sirius](https://sochisirius.ru/o-siriuse/programmes-of-the-educational-centre-sirius). The repo contains the scripts created by the participants of this program. They worked out the bioinformatics solution for the search of candidate genes connected with multifactorial diseases via the gene network analysis with algorithmic and machine learning approaches. 

## The scripts

* `src/khvorykh-g/run-svm-03.py` applies SVM (Support Vector Machine) to embedding of gene network to train the prognostic model and estimate the probablilitie of unlabeled data to belong to positive class (diasease genes).  

* `src/khvorykh-g/mask-genes-01.py` removes a random fraction of genes from the provided list of genes for the estimation of sensitivity of the bioinformatics pipeline.

* `src/khvorykh-g/estimate-quality-01.py` estimates the sensitivity of the method using the lists of masked genes (True Positives) and candidate genes.

* `src/khvorykh-g/generate-random-genes-02.py` generates random genes