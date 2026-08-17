# PCA Image Compression

A MATLAB project exploring **Principal Component Analysis (PCA) and Singular Value Decomposition (SVD)** for image compression and dimensionality reduction using the Extended Yale B face dataset.

## How It Works

- Constructed a high-dimensional data matrix from facial images
- Applied SVD to identify principal components and dominant image features
- Reconstructed images using different numbers of principal components
- Analyzed reconstruction error and compression performance
- Tested how PCA features generalize to unseen subjects
- Evaluated PCA's performance on a rotated image

## Technologies & Concepts

**MATLAB • PCA • SVD • Linear Algebra • Dimensionality Reduction • Image Compression • Data Analysis**

## Key Results

Using only **34 principal components** preserved 90% of the dataset's energy, reducing each image from 11,368 pixels to 34 coefficients. A 50-component representation achieved a **4.34% reconstruction error** on unseen subjects, while a rotated image produced a significantly higher error, demonstrating PCA's limitations with geometric transformations.

## What I Learned

This project strengthened my understanding of how **linear algebra can be used to reduce and analyze high-dimensional data**. I gained experience with PCA, SVD, low-rank approximations, image reconstruction, and evaluating how well learned features generalize to new data.

## Future Improvements

- Explore kernel PCA and other nonlinear dimensionality reduction methods
- Test additional image transformations
- Compare PCA with modern image compression techniques
