# Mining mechanical behaviors of gold nano-contact 

This repository hosts the code implementation of our analysis framework designed for the exploration of mechanical behaviors of gold nano-contacts, derived from Transmission Electron Microscopy (TEM) videos documenting the corresponding experiments. Utilizing in-situ TEM datasets collected during gold nano-contact experiments, our framework integrates a Generative Adversarial Network (GAN) - responsible for the synthesis of hypothetical contact images, and a Kernel Ridge Regression model - employed for the prediction of physical properties such as electrical conductance and the elastic spring constant, associated with these hypothetical contacts. A detailed breakdown of the repository's content is as follows:

* [ImgProcess](ImgProcess): This directory houses Python scripts which comprise modules implemented for the processing of TEM images. These scripts facilitate the detection of base regions, and the extraction of structural features from segmented nano-contact regions. Subsequently, these features are employed in the training of the Kernel Ridge Regression model (for physical property prediction), or input into the Metric Learning for Kernel Regression model, to learn an embedding representation.

* [GAN](GAN): This directory includes Python scripts used to train a Generative Adversarial Network (GAN). The GAN's role is to synthesize hypothetical nano-contacts, which are instrumental for the assembly of virtual experiments.

* [Unet](Unet): This directory contains Python scripts designed for training a U-net model. The function of this model is to automatically segment material regions by removing the background within TEM images.

The datasets used for training models, along with the corresponding results, are meticulously detailed and made available at the following link: https://zenodo.org/record/8036041.

