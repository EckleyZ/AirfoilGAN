# AirfoilGAN
Applying a General Adversarial Network (GAN) to an airfoil performance dataset to predict airfoil cross-sections based on required airfoil performance. 

Dataset of airfoil cross-sections was obtained from UIUC database at the link below:

UIUC database: https://m-selig.ae.illinois.edu/ads.html

Using this dataset each airfoil was analyzed at a large range of reynolds numbers and angles of attack. This performance data is currently begin fed through an autoencdoer (AE) to reduce the dimensionality of the input data. The encoded data is then passed into the GAN with some additional noise data to create unique airfoil cross-sections.

At this point in time there are a handful of issues:            
            
Markup :    - Quality of AE is poor due to the large amount of input data and small output dimension
            - Mode collapse is present in the generator 
            - Spikes appear in the loss curve
                        - Due to the mode collapse when the generator swtiches from one local solution to another there is a large increase in the loss as the generator transitions
                        - The spikes could also be due to a small batch size, a common problem with SGD where a batch has low diversity and results in a strange gradient
            - Steps are being taken to remove the spikes and reduce mode collapse in the model
                        - The addition of unrolled step in the generator has reduced the amount of mode collapse
                        - Changing the loss function of the discriminator to a wasseerstein loss function
                        - Improving the results from the AE through the use of a contractive AE or denoising AE
                        - Modifying how the dataset is being broken into chunks and fed to the AE and GAN
