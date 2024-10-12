# README

## Executive Summary

This paper analyzes the differences between the Preconditioned Steepest Descent (PSD) and Preconditioned Conjugate Gradient (PCG) methods, two foundational algorithms for efficiently solving large-scale linear systems and optimization problems in machine learning, engineering, and scientific computing. The study focuses on an \( n \times n \) symmetric positive definite (SPD) matrix \( A \) using three preconditioners: identity, Jacobi, and symmetric Gauss-Seidel. We evaluate the performance of PSD and PCG with different matrix generation techniques for \( n = 20 \) and \( n = 200 \), discussing convergence speeds and errors.

## Description of the Algorithms and Implementation

### Generating Matrix \( A \)

Two methods were employed to generate a symmetric positive definite matrix:
1. **Direct Symmetrization**: Random double-precision floating-point numbers between 0 and 1 are generated such that \( A_1[i, j] = A_1[j, i] \) for all \( i, j \). A constant of 10 is added to the diagonal.
2. **Cholesky-based Construction**: A lower triangular matrix \( L \) is generated with random values, and then \( A_2 \) is computed as \( A_2 = LL^T + 5I \). This ensures \( A_2 \) is SPD and improves numerical stability.

### Generating Preconditioners

Preconditioners were generated as follows:
- **Identity**: The simplest form.
- **Jacobi Preconditioner**: Defined as \( P_{\text{Jacobi}} = \text{diag}(A) \).
- **Symmetric Gauss-Seidel**: Given by \( P_{\text{SGS}} = (D-E)D^{-1}(D - E^T) \).

### Experimental Design and Results

During each iteration, values of \( ||r_k||_2 \) and relative error \( \frac{||x_k - x^*||_2}{||x^*||_2} \) were recorded. We plotted the logarithm of these errors against iteration numbers using R.

#### Results for Matrix \( A_1 \)

Using \( A_1 \) demonstrated that PSGS consistently outperformed other preconditioners. The results indicate that PSGS achieves faster convergence for both \( n = 20 \) and \( n = 200 \).

#### Results for Matrix \( A_2 \)

Similar trends were observed, though PSGS's performance varied based on matrix generation methods, highlighting the influence of matrix conditioning on convergence.

### Correctness Test

A correctness test using the matrix 
```math
A_{\text{test}} =
\begin{pmatrix}
5 & 7 & 6 & 5 \\
7 & 10 & 8 & 7 \\
6 & 8 & 11 & 9 \\
5 & 7 & 9 & 10
\end{pmatrix},
\quad
b_{\text{test}} =
\begin{pmatrix}
57 \\
79 \\
88 \\
86
\end{pmatrix}
```
The solution ```math
x = \begin{pmatrix} 1 \\ 2 \\ 3 \\ 4 \end{pmatrix}
``` was verified after convergence.

## Conclusion

This study emphasizes the importance of tailored approaches to solving linear systems. The choice of method and preconditioner should be informed by the properties of the matrix ```math
A
``` to achieve optimal performance.
