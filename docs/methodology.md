# How the experiment works

DNA methylation beta values describe how strongly a measured DNA site is methylated, on a scale from zero to one. The classifier tries to distinguish Normal samples from CIN3 samples. The small number of people makes overfitting a central concern.

Each split hides real samples before learning anything statistical. The training data determine which probes survive missingness checks, their imputation values, and the 50 CpGs with strongest ANOVA class association. A scaler and shrinkage linear discriminant classifier then learn from the training partition. Every method uses exactly the same selected features and real-fitted classifier scaler.

Random oversampling repeats real rows. SMOTE interpolates neighboring real rows of the same class. Diffusion learns to undo progressively added Gaussian noise while seeing the class and timestep, then starts from Gaussian noise and repeatedly denoises it to create synthetic examples. Its logit transform handles bounded beta values. Generated examples augment training only; every evaluated sample is real. Learning a distribution from a few dozen samples may fail, collapse or memorize.

The classifier decision boundary and empirical class priors may change with augmentation. All augmented approaches have equal final class counts, while real-only retains its observed class balance. This is part of the specified comparison. The generator's standardized logit space is separate from the classifier's standardized beta space.
