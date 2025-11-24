# HiFiGAN-ZH-16k Training

We trained the HiFiGAN-ZH-16k model using two datasets: NCSSD and AISHELL3. The total training time was as follows:

- **NCSSD**: 29.57 hours
- **AISHELL3**: 85 hours

The NCSSD dataset was resampled from dual-channel to single-channel audio, while the AISHELL3 dataset originally had a sampling rate of 44k, which was downsampled to 16k.

These datasets were used to train the HiFiGAN-ZH-16k model, resulting in high-quality audio synthesis for Mandarin speech.

**CheckPoint**: The training weights for steps 500k, 800k, 1000k, 1200k, 1300k, and 1500k are open-sourced and available for direct download. You can use them directly. For training steps beyond 1M, the reconstructed speech quality is excellent.
