# README: Analysis of Preconditioned Steepest Descent and Preconditioned Conjugate Gradient Methods

## Executive Summary
This paper explores the differences between two fundamental algorithms—Preconditioned Steepest Descent (PSD) and Preconditioned Conjugate Gradient (PCG)—which are crucial for efficiently solving large-scale linear systems and optimization problems across various domains, including machine learning and scientific computing. The study focuses on a square symmetric positive definite (SPD) matrix A and evaluates the effectiveness of three preconditioners: the identity, Jacobi, and symmetric Gauss-Seidel. Results are presented for matrix sizes n = 20 and n = 200, with a focus on performance metrics such as convergence speed and error analysis.

## Key Sections
1. **Description of the Algorithms and Implementation**:
   - **Matrix Generation**: Two methods are used to create SPD matrices: direct symmetrization and Cholesky-based construction.
   - **Preconditioner Generation**: The identity matrix, Jacobi, and symmetric Gauss-Seidel preconditioners are discussed in detail.
   - **Algorithm Overview**: Detailed implementations of both PSD and PCG methods, including initialization and iteration processes.

2. **Experimental Design and Results**:
   - The performance of the algorithms is evaluated through extensive trials, highlighting differences in convergence across various preconditioners.
   - Graphical representations and statistical summaries of iterations required for convergence are provided for both matrix generation methods.

3. **Correctness Test**:
   - The accuracy of the implemented algorithms is validated through a correctness test, ensuring solutions align with expected results.

4. **Conclusion**:
   - The study emphasizes the significant impact of matrix generation and preconditioner choice on convergence performance, highlighting the importance of tailored approaches in solving linear systems.

## Usage
This paper can serve as a resource for researchers and practitioners interested in numerical methods for solving linear systems. The findings illustrate the efficacy of preconditioners in enhancing convergence rates, which is especially relevant in fields requiring high-performance computing.
