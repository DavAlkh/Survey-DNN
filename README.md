# Survey-DNN
Implementation of deep neural network (DNN) in TensorFlow to estimate effect of galaxy survey window function on power spectrum and bispectrum.

It became a common practice to extract the cosmological information from galaxy redshift surveys using the the $n$-point statistics and their Fourier space counterparts, power spectrum (for 2 points) and bispectrum (for 3 points).

The imprint of the survey geometry generates the largest systematic discrepancy between the observed power spectrum, bispectrum and their theoretical predictions. Galaxy surveys cover only finite regions of space and thus the observed galaxy overdensity field $\delta_{\rm obs}(\mathbf{x})$ is affected by the survey geometry and weighting of tracers based on the selection criteria of the survey. If $W(\mathbf{x})$ is the survey window function, then $\delta_{\rm obs}(\mathbf{x})=W(\mathbf{x})\delta({\bf x})$, where $\delta({\bf x})$ is the ``true'' overdensity field, which is not affected by the survey systematic effects. Therefore, the observed power spectrum is
$$P_{\rm obs} ({\bf k}) = \int \frac{{\rm d}^3{q}}{(2\pi)^3} |\tilde{W}({\bf k}-{\bf q})|^2 P({\bf q})$$
    and bispectrum  
    $$B_{\rm obs} ({\bf k}_1,{\bf k}_2,{\bf k}_3)=\int  \frac{{\rm d}^3{q}}{(2\pi)^3}  \frac{{\rm d}^3{p}}{(2\pi)^3} \tilde{W}({\bf k}_1-{\bf q}) \tilde{W}({\bf k}_2-{\bf p}) \tilde{W}({\bf k}_3+{\bf q}+{\bf p}) B({\bf q},{\bf p},-{\bf q}-{\bf p})$$

Here we accelerate this numerically demanding operation using the deep neural networks. This speed-up is crucial when exploring the posterior distribution of cosmological parameters during MCMC sampling. The power spectrum DNN model is based on convolutional neural network and bispectum uses the U-Net model.
