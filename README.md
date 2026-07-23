# Contrastive Regulatory Embedding Attention Model (CREAM)
---

Contrastive Regulatory Embedding Attention Model (CREAM) is a computational framework that extends sequence-to-function (S2F) models with customized modules to better isolate cross-individual genetic variation and amplify causal eQTLs signals via an attention mechanism. CREAM can identify tissue-specific eQTLs and predict gene expression across diverse tissues. Furthermore, CREAM can provide predictive confidence (on *discrete* branch). We demonstrate that quantified predictive uncertainty serves as a proxy for prediction reliability. 


<p align="center">
  <img src="Figure0.pdf" alt="Model architecture" width="500">
</p>

The main features of CREAM include:
1)	Multiscale Feature Extraction & Contrastive Cross-Attention: CREAM mitigates downsampling resolution loss by establishing direct connections from intermediate convolutional layers to a customized contrastive cross-attention module. Given a pair of individual genomes, CREAM isolates multiscale genomic bins containing polymorphisms between individuals and contextualizes localized variant queries against global sequence backgrounds.
2)	eQTL-Guided Inductive Bias: We introduce an auxiliary loss function that aligns learned attention weights with statistically fine-mapped eQTL effect sizes. This regularization forces the network to prioritize functionally validated regulatory variants over non-causal background variation.
3)	Uncertainty Quantification: We reframe expression prediction as a hybrid discrete-continuous distribution task, leveraging Shannon entropy, predictive variance, and cross-run ensemble stability to track prediction reliability and quantify epistemic uncertainty.
