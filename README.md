## Computational Linear Algebra for Coders

This course is focused on the question: **How do we do matrix computations with acceptable speed and acceptable accuracy?**

This course is being taught in the [University of San Francisco's Masters of Science in Data Science](https://www.usfca.edu/arts-sciences/graduate-programs/data-science) program, summer 2018 (for graduate students studying to become data scientists).  The course is taught in Python with Jupyter Notebooks, using libraries such as Scikit-Learn and Numpy for most lessons, as well as Numba (a library that compiles Python to C for faster performance) and PyTorch (an alternative to Numpy for the GPU) in a few lessons.

You can find the 2017 version of the course [here](https://github.com/fastai/numerical-linear-algebra).

## Table of Contents
The following listing links to the notebooks in this repository, rendered through the [nbviewer](http://nbviewer.jupyter.org) service.  Topics Covered:
### [0. Course Logistics](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/0.%20Course%20Logistics.ipynb) 
  - [My background](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/0.%20Course%20Logistics.ipynb#Intro)
  - [Teaching Approach](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/0.%20Course%20Logistics.ipynb#Teaching)
  - [Importance of Technical Writing](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/0.%20Course%20Logistics.ipynb#Writing-Assignment)
  - [List of Excellent Technical Blogs](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/0.%20Course%20Logistics.ipynb#Excellent-Technical-Blogs)
  - [Linear Algebra Review Resources](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/0.%20Course%20Logistics.ipynb#Linear-Algebra)
  

### [1. Why are we here?](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/01-Why-are-we-here.ipynb) 
We start with a high level overview of some foundational concepts in numerical linear algebra.
  - [Matrix and Tensor Products](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/01-Why-are-we-here.ipynb#Matrix-and-Tensor-Products)
  - [Matrix Decompositions](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/01-Why-are-we-here.ipynb#Matrix-Decompositions)
  - [Accuracy](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/01-Why-are-we-here.ipynb#Accuracy)
  - [Memory use](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/01-Why-are-we-here.ipynb#Memory-Use)
  - [Speed](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/01-Why-are-we-here.ipynb#Speed)
  - [Parallelization & Vectorization](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/01-Why-are-we-here.ipynb#Vectorization)

### [2. Background Removal with SVD](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/02-Background-Removal-with-SVD.ipynb)
Another application of SVD is to identify the people and remove the background of a surveillance video.
 - [Load and View Video Data](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/02-Background-Removal-with-SVD.ipynb#Load-and-Format-the-Data)
  - [SVD](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/02-Background-Removal-with-SVD.ipynb#Singular-Value-Decomposition)
  - [Making a video](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/02-Background-Removal-with-SVD.ipynb#Make-Video)
  - [Speed of SVD for different size matrices](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/02-Background-Removal-with-SVD.ipynb#Speed-of-SVD-for-different-size-matrices)
  - [Two backgrounds in one video](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/02-Background-Removal-with-SVD.ipynb#2-Backgrounds-in-1-Video)
  - [Data compression](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/02-Background-Removal-with-SVD.ipynb#Aside-about-data-compression)
  
### [3. Topic Modeling with NMF and SVD](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/03-Topic-Modeling-with-NMF-and-SVD.ipynb) 
We will use the newsgroups dataset to try to identify the topics of different posts.  We use a term-document matrix that represents the frequency of the vocabulary in the documents.  We factor it using NMF and SVD, and compare the two approaches.
  - [Singular Value Decomposition](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/03-Topic-Modeling-with-NMF-and-SVD.ipynb#Singular-Value-Decomposition-(SVD))
  - [Non-negative Matrix Factorization (NMF)](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/03-Topic-Modeling-with-NMF-and-SVD.ipynb#Non-negative-Matrix-Factorization-(NMF))
  - [Topic Frequency-Inverse Document Frequency (TF-IDF)](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/03-Topic-Modeling-with-NMF-and-SVD.ipynb#TF-IDF)
  - [Stochastic Gradient Descent (SGD)](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/03-Topic-Modeling-with-NMF-and-SVD.ipynb#NMF-from-scratch-in-numpy,-using-SGD)
  - [Intro to PyTorch](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/03-Topic-Modeling-with-NMF-and-SVD.ipynb#PyTorch)
  - [Truncated SVD](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/03-Topic-Modeling-with-NMF-and-SVD.ipynb#Truncated-SVD)
  
### [4. Randomized SVD](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/04-Randomized-SVD.ipynb) 
  - [Random Projections with word vectors](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/04-Randomized-SVD.ipynb#Part-1:-Random-Projections-(with-word-vectors))
  - [Random SVD for Background Removal](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/04-Randomized-SVD.ipynb#Part-2:-Random-SVD-for-Background-Removal)
  - [Timing Comparison](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/04-Randomized-SVD.ipynb#Timing-Comparison)
  - [Math Details](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/04-Randomized-SVD.ipynb#Math-Details)
  - [Random SVD for Topic Modeling](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/04-Randomized-SVD.ipynb#Part-3:-Random-SVD-for-Topic-Modeling)

### [5. LU Factorization](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/05-LU-factorization.ipynb)
 - [Gaussian Elimination](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/05-LU-factorization.ipynb#Gaussian-Elimination)
 - [Stability of LU](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/05-LU-factorization.ipynb#Stability)
  - [LU factorization with Pivoting](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/05-LU-factorization.ipynb#LU-factorization-with-Partial-Pivoting)
  - [History of Gaussian Elimination](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/05-LU-factorization.ipynb#History-of-Gaussian-Elimination)
  - [Block Matrix Multiplication](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/05-LU-factorization.ipynb#Block-Matrices)

### [6. Compressed Sensing of CT Scans with Robust Regression](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/06-Compressed-Sensing-of-CT-Scans-with-Robust-Regression.ipynb)  
Compressed sensing is critical to allowing CT scans with lower radiation-- the image can be reconstructed with less data.  Here we will learn the technique and apply it to CT images.
  - [Broadcasting](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/06-Compressed-Sensing-of-CT-Scans-with-Robust-Regression.ipynb#Broadcasting)
  - [Sparse matrices](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/06-Compressed-Sensing-of-CT-Scans-with-Robust-Regression.ipynb#Sparse-Matrices-(in-Scipy))
  - [CT Scans and Compressed Sensing](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/06-Compressed-Sensing-of-CT-Scans-with-Robust-Regression.ipynb#Today:-CT-scans)
  - [L1 and L2 regression](https://nbviewer.jupyter.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/06-Compressed-Sensing-of-CT-Scans-with-Robust-Regression.ipynb#Regresssion)

### [7. Predicting Health Outcomes with Linear Regressions]() 
  - [Linear regression in sklearn]()
  - [Polynomial Features]()
  - [Speeding up with Numba]()
  - [Regularization and Noise]()

### [8. How to Implement Linear Regression]()
  - [How did Scikit Learn do it?]()
  - [Naive solution]()
  - [Normal equations and Cholesky factorization]()
  - [QR factorization]()
  - [SVD]()
  - [Timing Comparison]()
  - [Conditioning & Stability]()
  - [Full vs Reduced Factorizations]()
  - [Matrix Inversion is Unstable]()

### [9. PageRank with Eigen Decompositions]()
We have applied SVD to topic modeling, background removal, and linear regression. SVD is intimately connected to the eigen decomposition, so we will now learn how to calculate eigenvalues for a large matrix.  We will use DBpedia data, a large dataset of Wikipedia links, because here the principal eigenvector gives the relative importance of different Wikipedia pages (this is the basic idea of Google's PageRank algorithm).  We will look at 3 different methods for calculating eigenvectors, of increasing complexity (and increasing usefulness!).
  - [SVD]()
  - [DBpedia Dataset]()
  - [Power Method]()
  - [QR Algorithm]()
  - [Two-phase approach to finding eigenvalues]() 
  - [Arnoldi Iteration]()

### [10. Implementing QR Factorization]()
  - [Gram-Schmidt]()
  - [Householder]()
  - [Stability Examples]()

<hr>

## Why is this course taught in such a weird order?

This course is structured with a *top-down* teaching method, which is different from how most math courses operate.  Typically, in a *bottom-up* approach, you first learn all the separate components you will be using, and then you gradually build them up into more complex structures.  The problems with this are that students often lose motivation, don't have a sense of the "big picture", and don't know what they'll need.

Harvard Professor David Perkins has a book, [Making Learning Whole](https://www.amazon.com/Making-Learning-Whole-Principles-Transform/dp/0470633719) in which he uses baseball as an analogy.  We don't require kids to memorize all the rules of baseball and understand all the technical details before we let them play the game.  Rather, they start playing with a just general sense of it, and then gradually learn more rules/details as time goes on.

If you took the fast.ai deep learning course, that is what we used.  You can hear more about my teaching philosophy [in this blog post](http://www.fast.ai/2016/10/08/teaching-philosophy/) or [this talk I gave at the San Francisco Machine Learning meetup](https://vimeo.com/214233053).

All that to say, don't worry if you don't understand everything at first!  You're not supposed to.  We will start using some "black boxes" or matrix decompositions that haven't yet been explained, and then we'll dig into the lower level details later.

To start, focus on what things DO, not what they ARE.
