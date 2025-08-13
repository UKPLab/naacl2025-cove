# 5Pils-OOC 

5Pils-OOC is an extension of the [5Pils](https://github.com/UKPLab/5pils) dataset, published at EMNLP 2024.

It is a test set for two multimodal fact-checking tasks:
    🕵️ Context prediction (Image contextualization): predicting the true context of an image published on the web
    🧑‍⚖️ Veracity prediction: predicting whether a caption is accurate or out-of-context for an image


5Pils-OOC contains 624 images, each paired with an accurate caption and an out-of-context caption. All annotations have been extracted from articles written by fact-checking experts. 
The dataset is released under a **CC-BY-SA 4.0** license.
To download the images, please refer to the instructions of [5Pils](https://github.com/UKPLab/5pils).

## Structure

- *test.json*: the main file containing the true and false captions, and the URL to download the image
- *context_5pils_ooc_test.json*: the file containing the context labels for the context prediction task
- *evidence/direct_search*: the file containing the evidence retrieved by querying the Google search API with the image's caption


## Contact

- If you face troubles downloading some of the images (e.g., due to a broken URL link), please contact *jonathan.tonglet@tu-darmstadt.de*
