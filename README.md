# AirfoilGAN
Applying a General Adversarial Network (GAN) to an airfoil performance dataset to predict airfoil cross-sections based on required airfoil performance. 

Dataset of airfoil cross-sections was obtained from UIUC database at the link below:

UIUC database: https://m-selig.ae.illinois.edu/ads.html

Using this dataset each airfoil was analyzed at a large range of reynolds numbers and angles of attack. This performance data is currently begin fed through an autoencdoer to reduce the dimensionality of the input data. The encoded data is then passed into the GAN with some additional noise data to create unique airfoil cross-sections.

At this point in time there are a handful of issues:
Markup: * Quality of autoencoder is poor due to the large amount of input data and small output dimension
        * Small amount of mode collapse from the generator
        * Spikes appear in the loss curve during training:
              * The belief for right now is that because of the mode collapse in the generator it is switching between local solutions and causing large spikes in loss during the transition
              * The spikes could also be due to a small batch size
              * Steps are being taken to remove these spikes in the loss curve
