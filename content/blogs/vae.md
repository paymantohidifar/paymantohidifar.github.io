---
title: "Variational Autoencoders: Intuition and the Math Behind the Loss"
date: August 2026
description: A ground-up derivation of the VAE loss function, from Bayesian inference and the ELBO to the reparameterization trick and the reconstruction loss for both binary and continuous data.
tags:
  - Machine Learning
  - Generative Models
  - Deep Learning
---

My introduction to Variational Autoencoders (VAEs) didn't come from generating synthetic images, but from studying generative models that map protein sequences to functional fitness landscapes to predict mutational effects. VAEs evolved out of the conventional autoencoders that have been around for decades and were first introduced by Kingma & Welling in their 2013 paper *Auto-Encoding Variational Bayes*. Simply put, an autoencoder trains a neural network to reconstruct its inputs by compressing them through a lower-dimensional bottleneck. This compression forces the network to learn compact, meaningful representations, which is what makes traditional autoencoders such a powerful tool for tasks like dimensionality reduction, feature extraction, and anomaly detection.

<figure>
  <img src="/static/assets/autoencoder.png" width=500px>
  <figcaption>A standard autoencoder: the encoder compresses the input into a fixed point in latent space, and the decoder reconstructs it from that point.</figcaption>
</figure>

Standard autoencoders make poor generative models, though, because the encoder maps inputs to fixed, isolated points in that lower-dimensional space (the "latent space"). Since each input lands on a fixed point, sampling or interpolating between those isolated points and feeding the result to the decoder may not produce anything meaningful. VAEs fix this by making the encoder output a probability distribution over the latent space rather than a single static vector. Instead of deterministically mapping an input to a point, the encoder outputs a mean $\boldsymbol{\mu}$ and variance $\boldsymbol{\sigma}^2$, essentially saying, "this input belongs in a region centered here, with this degree of uncertainty". During training, we push all of these distributions to look Gaussian and to overlap with one another in latent space, forming a smooth, continuous "manifold" that the decoder can meaningfully sample and interpolate across to produce novel outputs that still resemble the training examples.

The training procedure for VAEs is well-established at this point, but it's worth understanding the underlying assumptions that go into formulating the loss function we're actually minimizing. In this post, I'll walk through that derivation step by step and try to build intuition for each piece along the way. I've also put together a short notebook that trains a convolutional VAE on the Fashion-MNIST dataset using PyTorch for anyone who wants to see it in code: [`notebooks/autoencoders.ipynb`](notebooks/autoencoders.ipynb).

## The Statistical Motivation

From a statistics perspective, a VAE is a Bayesian inference problem. Say we have an observed dataset $x$ (e.g. a set of molecules, protein sequences, or images) that we believe was generated from some latent variables $z$ that are unknown to us. We want to learn the distribution over $z$ that our data was generated from. In other words, we'd like the posterior distribution $p(z \mid x)$. By Bayes' theorem:

$$p(z \mid x) = \frac{p(x \mid z)\, p(z)}{p(x)}$$

<figure>
  <img src="/static/assets/vae-graph.png" width=500px>
  <figcaption>The generative process: latent variable $z$ gives rise to observed data $x$, and we want to infer $z$ from $x$ by inverting this relationship.</figcaption>
</figure>

where $p(z)$ is the prior distribution over the latent variable, and $p(x)$ is the marginal likelihood of the data — obtained by marginalizing the joint distribution $p(x,z)$ over $z$. $p(x)$ can be computed by integrating over the entire continuous latent space (or summed, if the latent space is discrete):

$$p(x) = \int_z p(x,z)\,dz = \int_z p(x\mid z)\,p(z)\,dz$$

Computing $p(x)$ in the denominator is generally intractable (difficult or impossible to compute). So instead of computing the true posterior $p(z \mid x)$ exactly, we approximate it with a tractable distribution $q_{\theta}(z \mid x)$ and try to make $q_{\theta}(z \mid x)$ as close as possible to the true $p(z \mid x)$. "As close as possible" needs a similarity measure between distributions, and the standard choice is the Kullback-Leibler (KL) divergence. Here $\theta$ denotes the set of parameters governing the approximate posterior.

## Kullback-Leibler (KL) Divergence

There are a few ways to motivate the KL divergence; one of the more intuitive routes is through information theory. The core idea is that the more probable an event is, the less "information" it carries — if someone tells you something you already expected, they haven't given you much new information. This inverse relationship between probability and information is modeled as:

$$\text{I}_p(x) = -\log p(x)$$

If we compare two distributions $q(x)$ and $p(x)$, the difference in information they assign to the same event is:

$$-\log p(x) - \big(-\log q(x)\big) = \log \frac{q(x)}{p(x)}$$

The KL divergence is just the expectation of that difference, taken with respect to one of the distributions:

$$D_{KL}\big(q(x)\,\|\,p(x)\big) = \mathbb{E}_{\sim q(x)}\left[\log \frac{q(x)}{p(x)}\right] = \int q(x) \log \frac{q(x)}{p(x)} dx$$

Because the expectation is taken with respect to $q$, swapping the order of $p$ and $q$ gives a different value. In other words, it's not symmetric:

$$D_{KL}(q\|p) \ne D_{KL}(p\|q)$$

It's also always non-negative, $D_{KL}(q\|p) \ge 0$. We can show this using the inequality $\log t \le t - 1$. Applying it with $t = p(x)/q(x)$ inside the KL expression and simplifying shows the expression can never go negative:

$$-D_{KL}\big(q(x)\,\|\,p(x)\big) = \int q(x) \log \frac{p(x)}{q(x)}dx \le$$
$$\int q(x) \left(\frac{p(x)}{q(x)}-1\right)dx=$$
$$\int p(x)dx - \int q(x)dx = 1 - 1 = 0$$
$$D_{KL}\big(q(x)\,\|\,p(x)\big) \ge 0$$

This one fact — non-negativity — is what makes the whole variational inference trick work, as we'll see next.

## From KL Divergence to the ELBO

Now let's go back to our original problem: minimizing the difference between $p(z \mid x)$ and its proxy $q_\theta(z\mid x)$. Given a dataset $\mathcal{D} = \{x_1, x_2, \dots, x_N\}$ of $N$ datapoints, we can write the KL divergence for a single datapoint $x_i$ as:

$$D_{KL}\big(q_\theta(z \mid x_i)\,\|\,p(z \mid x_i)\big) = \mathbb{E}_{\sim q_\theta(z\mid x_i)}\left[\log \frac{q_\theta(z \mid x_i)}{p(z \mid x_i)}\right] \ge 0$$
$$-\int q_{\theta}(z \mid x_i)\log \frac{p(z \mid x_i)}{q_{\theta}(z \mid x_i)} dz \ge 0$$

Using Bayes' theorem:

$$-\int q_{\theta}(z \mid x_i) \log \frac{p(x_i \mid z)p(z)}{q_{\theta}(z \mid x_i)p(x_i)}dz \ge 0$$
$$-\int q_{\theta}(z \mid x_i)\left[ \log \frac{p(z)}{q_{\theta}(z \mid x_i)} + \log p(x_i \mid z) - \log p(x_i) \right] dz \ge 0$$
$$\int q_{\theta}(z \mid x_i)\log \frac{q_{\theta}(z \mid x_i)}{p(z)} dz - \int q_{\theta}(z \mid x_i) \log p(x_i \mid z) dz + \int q_{\theta}(z \mid x_i) \log p(x_i) dz \ge 0$$
$$D_{KL}\big(q_\theta(z \mid x_i)\,\|\,p(z)\big) - \int q_{\theta}(z \mid x_i) \log p(x_i \mid z) dz + \int q_{\theta}(z \mid x_i) \log p(x_i) dz \ge 0$$

Because $\log p(x_i)$ doesn't depend on $z$, it comes straight out of the integral as a constant; and since $q_\theta(z\mid x_i)$ integrates to 1, that term just leaves $\log p(x_i)$ behind untouched:

$$\log p(x_i) \ge -D_{KL}\big(q_\theta(z \mid x_i)\,\|\,p(z)\big) + \mathbb{E}_{\sim q_{\theta}(z \mid x_i)} \left[\log p(x_i \mid z)\right]$$

We also assume the generative distribution is specified by a separate set of parameters $\phi$, i.e. $p_\phi(x_i \mid z)$; we'll get to $\theta$ and $\phi$ in more depth below. The final form can be written as:

$$\log p(x_i) \ge \mathbb{E}_{\sim q_{\theta}(z \mid x_i)} \left[\log p_\phi(x_i \mid z)\right] - D_{KL}\big(q_\theta(z \mid x_i)\,\|\,p(z)\big)$$

The right-hand side is the **Evidence Lower BOund (ELBO)**, also called the variational lower bound. We'd like to maximize $\log p(x_i)$, the log-likelihood of our data, directly, but that's intractable because it depends on the same troublesome integral as before. The ELBO gives us a lower bound on that quantity that we can compute and optimize instead. That's a maximization problem, but we can turn it into a minimization problem by defining the loss function as the negative of the ELBO:

$$\mathcal{L}(x_i; \theta, \phi) = -\mathbb{E}_{\sim q_{\theta}(z \mid x_i)} \left[\log p_\phi(x_i \mid z)\right] + D_{KL}\big(q_\theta(z \mid x_i)\,\|\,p(z)\big)$$

In this loss function, the first term, $-\mathbb{E}_{q_\theta(z\mid x_i)}\big[\log p_\phi(x_i \mid z)\big]$, is the reconstruction loss. It measures how well the generative model (also called the decoder) can reproduce $x_i$ given a latent sample $z$ drawn from the encoder's distribution. The second term, $D_{KL}\big(q_\theta(z\mid x_i)\,\|\,p(z)\big)$, is the regularization term. It constrains the encoder's output distribution to stay close to a chosen prior $p(z)$, which is what keeps the latent space smooth and well-behaved rather than collapsing into disconnected point estimates.

This is precisely the tension described from the architectural side above: reconstruct the input faithfully, but keep the latent representation "spread out" and well-structured rather than memorizing brittle point encodings.

## Choosing Gaussian Latents

One way to minimize the loss objective is to use a Monte Carlo approximation of the loss gradients, but this is computationally expensive for large datasets. Instead, we parameterize both the encoder and decoder with neural networks and minimize the loss via backpropagation. To train the network this way, we need a loss we can compute (and differentiate) in closed form wherever possible. The standard choice, and the one Kingma & Welling used in the original VAE paper, is to let both the prior and the approximate posterior be Gaussian:

$$p(z) = \mathcal{N}(0, I), \qquad q_\theta(z \mid x_i) = \mathcal{N}\big(\mu^{(i)},\, \sigma^{(i)2} I\big)$$

With this choice, the KL term in the ELBO has an exact closed-form solution. Working through the Gaussian entropy and cross-entropy terms — expanding the log ratio of two Gaussian densities, using the definition of variance, and simplifying — collapses the regularization term for a $J$-dimensional latent vector to:

$$D_{KL}\big(q_\theta(z\mid x_i)\,\|\,p(z)\big) = -\frac{1}{2}\sum_{j=1}^{J}\Big(1 + \log \sigma^{(i)2}_j - \mu^{(i)2}_j - \sigma^{(i)2}_j\Big)$$

No sampling or Monte Carlo estimate is needed for this piece — it's an exact function of the mean and variance vectors the encoder outputs. This is the well-known "KL term" you'll see in essentially every VAE implementation.

The reconstruction term, by contrast, generally does need to be estimated by sampling. In other words, we draw $z$ from $q_\theta(z\mid x_i)$ and evaluate $\log p_\phi(x_i \mid z)$ through the decoder. But sampling is a stochastic operation, and stochastic nodes aren't differentiable, which would break backpropagation during training. To fix that, we need one more trick.

## The Reparameterization Trick

You can't directly backpropagate gradients through a stochastic node, meaning that "sample $z$ from $\mathcal{N}(\mu, \sigma^2)$" isn't differentiable with respect to $\mu$ and $\sigma$ in any useful way. To fix this, we use the reparameterization trick: instead of sampling $z$ directly, we sample a noise variable $\epsilon$ from a fixed, parameter-free standard normal, and construct $z$ as a deterministic, differentiable function of $\mu$, $\sigma$, and $\epsilon$:

$$\epsilon \sim \mathcal{N}(0, I), \qquad z = \mu + \sigma \odot \epsilon$$

Now the randomness is isolated in $\epsilon$, which requires no gradient, while $\mu$ and $\sigma$, the actual outputs of the encoder network, sit on a fully differentiable path. Gradients flow through them normally during backpropagation, and the network learns to shape the mean and variance of its latent distributions just like it would learn any other parameters.

<figure>
  <img src="/static/assets/reparameterization.png" width=500px>
  <figcaption>The reparameterization trick reroutes the stochastic sampling step through a fixed noise source $\epsilon$, keeping the path from $\mu$ and $\sigma$ to $z$ differentiable.</figcaption>
</figure>

## Putting It Together: The Final Loss

Going back to the loss objective, the reconstruction term is mathematically intractable in general. To handle it, we approximate the expectation with Monte Carlo integration. Instead of integrating over the whole distribution, we draw $L$ random samples of $z$ from the encoder's distribution $q_\theta(z\mid x_i)$:

$$\mathbb{E}_{q_\theta(z\mid x_i)}\big[\log p_\phi(x_i \mid z)\big] \approx \frac{1}{L} \sum_{l=1}^L \log p_\phi(x_i \mid z_l)$$

In the original VAE paper, Kingma & Welling showed that if your mini-batch size is large enough (e.g., 32, 64, or 128 datapoints), setting $L = 1$ (drawing just one random latent sample $z$ per input) provides an unbiased estimate with low enough variance for stable gradient descent. Because we repeat this across a mini-batch of many inputs, the average loss over the batch effectively serves as the expectation operator.

Combining the closed-form KL term with a single-sample Monte Carlo estimate of the reconstruction term, the quantity to maximize for a single datapoint $x_i$ and a single stochastic draw becomes:

$$\mathcal{G}(x_i;\theta,\phi) = \log p_\phi(x_i \mid z) - \frac{1}{2}\sum_{j=1}^{J}\Big(1 + \log \sigma^{(i)2}_j - \mu^{(i)2}_j - \sigma^{(i)2}_j\Big)$$

As mentioned earlier, since training frameworks minimize rather than maximize, we flip the sign to get the actual loss for that datapoint:

$$\mathcal{L}(x_i;\theta,\phi) = -\log p_\phi(x_i \mid z) + \frac{1}{2}\sum_{j=1}^{J}\Big(1 + \log \sigma^{(i)2}_j - \mu^{(i)2}_j - \sigma^{(i)2}_j\Big)$$

The loss for the entire dataset is then simply the sum (or, in practice, the average within a mini-batch) of this per-datapoint loss over all $N$ datapoints:

$$\mathcal{L}(\theta,\phi) = \sum_{i=1}^{N} \mathcal{L}(x_i;\theta,\phi)$$

There's a second, separate independence assumption worth unpacking inside the reconstruction term itself. Suppose each datapoint has $D$ dimensions (e.g., $D$ pixels in an image, or $D$ amino acids in a protein sequence). Given the latent code $z$, we assume the individual feature dimensions of $x_i$ are conditionally independent of one another, so the reconstruction term factorizes into a sum of per-dimension log-likelihoods:

$$\log p_\phi(x_i \mid z) = \sum_{k=1}^{D} \log p_\phi(x_{i,k} \mid z)$$

In practice, this per-dimension term is often implemented as element-wise binary cross-entropy or mean squared error, depending on whether the task is classification or regression. The second term above is the closed-form Gaussian KL penalty derived earlier. Training the VAE means finding encoder and decoder parameters $(\theta^*, \phi^*)$ that jointly minimize $\mathcal{L}(\theta,\phi)$ over the whole dataset.

In the following section, we'll derive the reconstruction loss for a single datapoint based on the model task, dropping the datapoint index $i$ for brevity and writing $x$ for that one datapoint's $D$-dimensional feature vector.

### Reconstruction Loss for Binary Classification

We can model a binary feature $x_k$ (e.g. an image pixel that is either 0 or 1) with a Bernoulli distribution. The decoder network outputs a reconstructed probability $\hat x_k = f_\phi(z)_k$ for $x_k = 1$ using a sigmoid activation function. The Probability Mass Function (PMF) for a single Bernoulli random variable is:

$$p_\phi(x_k \mid z) = \hat x_k^{\,x_k}(1 - \hat x_k)^{1 - x_k}$$

After applying the log operator to the Bernoulli PMF for a single feature:

$$\log p_\phi(x_k \mid z) = x_k \log \hat x_k + (1 - x_k) \log (1 - \hat x_k)$$

Since the reconstruction loss is defined as $-\log p_\phi(x_k \mid z)$, negating both sides gives $-\big[x_k \log \hat x_k + (1 - x_k) \log (1 - \hat x_k)\big]$ — exactly the standard binary cross-entropy loss. Summed over all $D$ pixels, this is the reconstruction loss implemented in practice.

The same logic applies to multi-class classification tasks, where the loss objective is computed with the categorical cross-entropy loss function.

### Reconstruction Loss for Continuous Regression

To understand why Mean Squared Error (MSE) is used as a reconstruction loss, we look back at the core VAE objective: maximizing the log-likelihood $\log p_\phi(x \mid z)$. While BCE assumes our data follows a Bernoulli distribution, MSE emerges naturally when we assume our data follows a Gaussian distribution with a fixed variance. This makes MSE the mathematically appropriate choice for unconstrained continuous data.

Again, suppose our continuous datapoint $x$ has $D$ dimensions. We assume that given the latent code $z$, each feature $x_k$ is normally distributed around a mean $\mu_k$ predicted by the decoder, with some fixed variance $\sigma^2$:

$$x_k \mid z \sim \mathcal{N}(\mu_k, \sigma^2)$$

The Probability Density Function (PDF) for a single Gaussian random variable is:

$$p_\phi(x_k \mid z) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(x_k - \mu_k)^2}{2\sigma^2} \right)$$

Here, the decoder network outputs the mean $\mu_k = f_\phi(z)_k$ (usually using a linear or identity activation function so the values can be any real number).

Assuming independence across all $D$ dimensions, the joint probability is the product of the individual per-dimension densities. Taking the natural log converts the product into a sum:

$$\log p_\phi(x \mid z) = \sum_{k=1}^{D} \log p_\phi(x_k \mid z)$$

Now we plug the Gaussian PDF into the log-likelihood function:

$$\log p_\phi(x_k \mid z) = \log \left( \frac{1}{\sqrt{2\pi\sigma^2}} \right) + \log \left( \exp\left( -\frac{(x_k - \mu_k)^2}{2\sigma^2} \right) \right)$$

Using log properties ($\log(\exp(A)) = A$), this simplifies to:

$$\log p_\phi(x_k \mid z) = -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(x_k - \mu_k)^2}{2\sigma^2}$$

To get the final reconstruction loss, we take the negative of the total log-likelihood:

$$\mathcal{L}_{\text{recon}} = -\sum_{k=1}^{D} \log p_\phi(x_k \mid z) = \sum_{k=1}^{D} \left[ \frac{1}{2}\log(2\pi\sigma^2) + \frac{(x_k - \mu_k)^2}{2\sigma^2} \right]$$

When training a neural network, any term that doesn't depend on the network's weights $\phi$ can be dropped, since its derivative with respect to the weights is zero.

* The term $\frac{1}{2}\log(2\pi\sigma^2)$ is a constant.
* If we assume a fixed standard deviation of $\sigma = \frac{1}{\sqrt{2}}$ (or treat $\frac{1}{2\sigma^2}$ as a constant scaling factor), the denominator simplifies.

Dropping these constants leaves us with the core optimization term:

$$\mathcal{L}_{\text{recon}} \propto \sum_{k=1}^{D} (x_k - \mu_k)^2$$

This is the exact formula for Sum of Squared Errors, which is simply a scaled version of Mean Squared Error (MSE).

## Why This Actually Produces a Useful Latent Space

Tying the math back to the original motivation: the KL term is what makes the smooth, continuous latent space possible. By constantly pulling every encoded distribution toward a shared standard normal prior, training prevents the encoder from carving out isolated, disconnected regions for each input — instead, those regions overlap and blend, since they're all being pulled toward the same central distribution. That overlap is exactly why interpolating between two points in a VAE's latent space produces a plausible sequence of intermediate outputs, and why sampling directly from the prior $\mathcal{N}(0, I)$ and feeding it to the decoder produces coherent, novel samples rather than noise.

Meanwhile, the reconstruction term keeps the encoder honest — it can't just collapse everything to the same point in latent space to minimize the KL penalty, because doing so would make reconstruction impossible. The loss function balances these two constraints, and the reparameterization trick is what makes that balance trainable end-to-end with ordinary gradient descent.

## Summary

* A regular autoencoder's latent space isn't guaranteed to be continuous or well-structured; a VAE fixes this by encoding *distributions* instead of points.
* This architectural choice is really an approximate Bayesian inference problem: we can't compute the true posterior $p(z\mid x)$ directly, so we approximate it with $q_\theta(z\mid x)$ and minimize their KL divergence.
* Because that KL divergence is itself intractable to minimize directly, we instead maximize the ELBO — a computable lower bound on the data log-likelihood — which is mathematically equivalent to minimizing the KL divergence we actually care about.
* The ELBO splits cleanly into a reconstruction term and a KL regularization term; choosing Gaussian priors and posteriors gives the KL term a closed form.
* The reparameterization trick ($z = \mu + \sigma \odot \epsilon$) makes the sampling step differentiable, allowing standard backpropagation to train both encoder and decoder end-to-end.

## References

1. Kingma, D. P., & Welling, M. (2013). [*Auto-Encoding Variational Bayes*](https://arxiv.org/abs/1312.6114). arXiv:1312.6114.
2. Jordan, J. [*Variational Autoencoders*](https://www.jeremyjordan.me/variational-autoencoders/).
