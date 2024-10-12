// FCM_HW3.cpp : This file contains the 'main' function. Program execution begins and ends there.

#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <numeric>

std::vector<std::vector<double>> matrixMatrixMultiplication(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B) {
    int numRowsA = A.size();
    int numColsA = A[0].size();
    int numRowsB = B.size();
    int numColsB = B[0].size();

    if (numColsA != numRowsB) {
        std::cout << "Matrix dimensions are incompatible for multiplication." << std::endl;
        return std::vector<std::vector<double>>();
    }

    std::vector<std::vector<double>> result(numRowsA, std::vector<double>(numColsB, 0));

    for (int i = 0; i < numRowsA; i++) {
        for (int j = 0; j < numColsB; j++) {
            for (int k = 0; k < numColsA; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return result;
}

std::vector<std::vector<double>> transposeMatrix(const std::vector<std::vector<double>>& matrix) {
    int numRows = matrix.size();
    int numCols = matrix[0].size();
    std::vector<std::vector<double>> transpose(numCols, std::vector<double>(numRows, 0.0));

    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            transpose[j][i] = matrix[i][j];
        }
    }

    return transpose;
}

std::vector<double> generateVectorB(int n) {
    std::vector<double> b(n, 0.0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    for (int i = 0; i < n; i++) {
        b[i] = dis(gen);
    }

    return b;
}

std::vector<double> matrixVectorMultiply(const std::vector<std::vector<double>>& A, const std::vector<double>& x) {
    int numRows = A.size();
    int numCols = A[0].size();  // Assuming all columns have the same size

    std::vector<double> Ax(numRows, 0);

    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            Ax[i] += A[i][j] * x[j];
        }
    }

    return Ax;
}

std::vector<std::vector<double>> matrixSubtraction(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B) {
    int numRowsA = A.size();
    int numColsA = A[0].size();
    int numRowsB = B.size();
    int numColsB = B[0].size();

    if (numRowsA != numRowsB || numColsA != numColsB) {
        std::cout << "Matrix dimensions are incompatible for subtraction." << std::endl;
        return std::vector<std::vector<double>>();
    }

    std::vector<std::vector<double>> result(numRowsA, std::vector<double>(numColsA, 0));

    for (int i = 0; i < numRowsA; i++) {
        for (int j = 0; j < numColsA; j++) {
            result[i][j] = A[i][j] - B[i][j];
        }
    }

    return result;
}

std::vector<double> vectorSubtraction(const std::vector<double>& a, const std::vector<double>& b) {
    // Ensure that both vectors have the same size
    if (a.size() != b.size()) {
        // Handle the error appropriately; here, I'm throwing an exception
        throw std::invalid_argument("Vectors must have the same size for subtraction.");
    }

    int n = a.size();
    std::vector<double> result(n);

    for (int i = 0; i < n; ++i) {
        result[i] = a[i] - b[i];
    }

    return result;
}


std::vector<double> vectorAddition(const std::vector<double>& vector1, const std::vector<double>& vector2) {
    int size1 = vector1.size();
    int size2 = vector2.size();

    // Check if the sizes of the input vectors match
    if (size1 != size2) {
        throw std::runtime_error("Vector sizes do not match for addition.");
    }

    std::vector<double> result(size1);

    for (int i = 0; i < size1; i++) {
        result[i] = vector1[i] + vector2[i];
    }

    return result;
}

double dotProduct(const std::vector<double>& r, const std::vector<double>& z) {
    if (r.size() != z.size()) {
        throw std::runtime_error("Vectors must have the same size for dot product calculation.");
    }

    double result = 0.0;
    for (size_t i = 0; i < r.size(); i++) {
        result += r[i] * z[i];
    }
    return result;
}

std::vector<double> scalarVectorMultiplication(double scalar, const std::vector<double>& vector) {
    std::vector<double> result(vector.size(), 0.0);

    for (size_t i = 0; i < vector.size(); i++) {
        result[i] = scalar * vector[i];
    }

    return result;
}

double vectorTwoNorm(const std::vector<double>& x) {
    double norm = 0.0;
    for (int i = 0; i < x.size(); i++) {
        norm += std::pow(x[i], 2);
    }
    return std::sqrt(norm);
}

std::vector<double> J_find_zk(const std::vector<std::vector<double>>& P, const std::vector<double>& b) {
    int n = P.size();
    std::vector<double> z_k(n, 0.0);

    for (int i = 0; i < n; i++) {
        if (P[i][i] != 0.0) {
            z_k[i] = b[i] / P[i][i];
        }
        else {
            std::cout << "Error: division by zero when solving for z_k in J_find_zk()" << std::endl;
        }
    }

    return z_k;
}

// -------------------- Generating A ------------------------

std::vector<std::vector<double>> generateLowerTriangularMatrix(int n) {
    std::vector<std::vector<double>> L(n, std::vector<double>(n, 0.0));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution < double > dis(0.0, 1.0); 

    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            L[i][j] = dis(gen);
        }
    }
    return L;
}

std::vector<std::vector<double>> alternativeSPD(int n) {
    
    std::vector<std::vector<double>> L = generateLowerTriangularMatrix(n);
    std::vector<std::vector<double>> LT = transposeMatrix(L);
    std::vector<std::vector<double>> LLT = matrixMatrixMultiplication(L, LT);

    for (int i = 0; i < n; i++) {
        LLT[i][i] += 5.0;
    }

    return LLT;
}

std::vector<std::vector<double>> generateSPD(int n) {
    std::vector < std::vector < double >> A(n, std::vector<double>(n, 0.0));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution < double > dis(0.0, 1.0);

    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            double value = dis(gen);
            // makes it symmetric
            A[i][j] = value;
            A[j][i] = value; 
        }
    }

    // Add a small positive constant to the diagonal elements
    for (int i = 0; i < n; i++) {
        A[i][i] += 10;
    }

    return A;
}

// ----------- Preconditioner generation -------------------------------

std::vector<std::vector<double>> generateIdentityPreconditioner(int n) {
    std::vector<std::vector<double>> P_I(n, std::vector<double>(n, 0.0));
    
    for (int i = 0; i < n; i++) {
        P_I[i][i] = 1;
    }
    return P_I;
}

std::vector<std::vector<double>> generateJacobiPreconditioner(const std::vector<std::vector<double>>& A) {
    int n = A.size();
    std::vector<std::vector<double>> P(n, std::vector<double>(n, 0.0));

    for (int i = 0; i < n; i++) {
        P[i][i] = A[i][i]; 
    }

    return P;
}

std::vector<std::vector<double>> generateC(const std::vector<std::vector<double>>& A) {
    int n = A.size();
    std::vector<std::vector<double>> D(n, std::vector<double>(n, 0.0));
    //create D
    for (int i = 0; i < n; i++) {
        D[i][i] = A[i][i];
    }

    //create E
    std::vector<std::vector<double>> E(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            E[i][j] = -A[i][j];
        }
    }

    //create D^-1/2
    std::vector<std::vector<double>> D_half(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; i++) {
        D_half[i][i] = 1 / std::sqrt(D[i][i]);
    }

    //create C
    std::vector<std::vector<double>> C = matrixMatrixMultiplication(matrixSubtraction(D, E), D_half); 

    /*std::cout << "Matrix C: " << std::endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << C[i][j] << " ";
        }
        std::cout << std::endl;
    }*/

    return C;
}

std::vector<std::vector<double>> generateCT(const std::vector<std::vector<double>>& A) {
    int n = A.size();
    std::vector<std::vector<double>> D(n, std::vector<double>(n, 0.0));
    //create D
    for (int i = 0; i < n; i++) {
        D[i][i] = A[i][i];
    }

    //create E
    std::vector<std::vector<double>> E(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            E[i][j] = -A[i][j];
        }
    }

    //create D^-1/2
    std::vector<std::vector<double>> D_half(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; i++) {
        D_half[i][i] = 1 / std::sqrt(D[i][i]);
    }

    //create C
    std::vector<std::vector<double>> CT = matrixMatrixMultiplication(D_half, matrixSubtraction(D, transposeMatrix(E)));

    /*std::cout << "Matrix C^T: " << std::endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << CT[i][j] << " ";
        }
        std::cout << std::endl;
    }*/

    return CT;
}

std::vector<double> forwardSubstitution(const std::vector<std::vector<double>>& L, const std::vector<double>& b) {
    int n = L.size();
    std::vector<double> y(n, 0);

    for (int i = 0; i < n; i++) {
        if (L[i][i] == 0) {
            std::cout << "ERROR: Det(A) is nonzero" << std::endl;
        }
        else {
            y[i] = b[i];
            for (int j = 0; j < i; j++) {
                y[i] -= L[i][j] * y[j];
            }
            y[i] /= L[i][i];
        }
    }

    return y;
}

std::vector<double> backwardSubstitution(const std::vector<std::vector<double>>& U, const std::vector<double>& y) {
    int n = U.size();
    std::vector<double> x_tilde(n, 0);

    for (int i = n - 1; i >= 0; i--) {
        if (U[i][i] == 0) {
            std::cout << "ERROR: Det(A) is nonzero" << std::endl;
        }
        else {
            x_tilde[i] = y[i];
            for (int j = n - 1; j > i; j--) {
                x_tilde[i] -= U[i][j] * x_tilde[j];
            }
            x_tilde[i] /= U[i][i];
        }

    }

    return x_tilde;
}

// ------------------ Steepest Descent Methods ----------------------------

std::vector<double> IsteepestDescent(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& P, const std::vector<double>& b) {
    std::vector<double> x_star = { 1, 2, 3, 4 };
    int n = A.size();
    std::vector<double> x_k(n, 0.0); //x_0 all 0s
    // r_0 = b - Ax_0.. implies r0 = b
    std::vector<double> r_k = b;
    // Solve Pz_0 = r_0.. implies z0 = r0 = b
    std::vector<double> z_k = r_k;

    std::ofstream csvFile("fcm_data1.csv");
    csvFile << "Iteration,ResidualNorm,r/b\n";

    for (int k = 1; k <= 10000; k++) {
        std::vector<double> omega_k = matrixVectorMultiply(A, z_k);
        double alpha_k = dotProduct(z_k, r_k) / dotProduct(z_k, omega_k);
        x_k = vectorAddition(x_k, scalarVectorMultiplication(alpha_k, z_k));
        r_k = vectorSubtraction(r_k, scalarVectorMultiplication(alpha_k, omega_k));
        z_k = r_k;

        double r_kTwoNorm = vectorTwoNorm(r_k);
        //std::cout << "Iteration " << k << ": Residual Norm = " << r_kTwoNorm << std::endl;
        double rk_over_b = r_kTwoNorm / vectorTwoNorm(b);
        double x_min_xstar_twoNorm = vectorTwoNorm(vectorSubtraction(x_k, x_star)) / vectorTwoNorm(x_star);
        csvFile << k << "," << r_kTwoNorm << "," << x_min_xstar_twoNorm << "\n";
        std::cout << "Iteration " << k << ": Residual Norm = " << r_kTwoNorm << " r/b: " << rk_over_b << std::endl;
        if (r_kTwoNorm < 1e-6) {
            std::cout << "Converged after " << k << " iterations." << std::endl;
            break;
        }
    }
    return x_k;
}

std::vector<double> JsteepestDescent(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& P, const std::vector<double>& b) {
    std::vector<double> x_star = { 1, 2, 3, 4 };
    int n = A.size();
    std::vector<double> x_k(n, 0.0); //x_0 all 0s
    // r_0 = b - Ax_0.. implies r0 = b
    std::vector<double> r_k = b;
    // Solve Pz_0 = r_0.. implies z_i = b_i/a_ii
    std::vector<double> z_k = J_find_zk(P, b);

    std::ofstream csvFile("fcm_data2.csv");
    csvFile << "Iteration,ResidualNorm,RelativeError\n";
   
    for (int k = 1; k <= 10000; k++) {
        std::vector<double> omega_k = matrixVectorMultiply(A, z_k);
        double alpha_k = dotProduct(z_k, r_k) / dotProduct(z_k, omega_k);
        x_k = vectorAddition(x_k, scalarVectorMultiplication(alpha_k, z_k));
        r_k = vectorSubtraction(r_k, scalarVectorMultiplication(alpha_k, omega_k));
        z_k = J_find_zk(P, r_k);

        double r_kTwoNorm = vectorTwoNorm(r_k);
        double x_min_xstar_twoNorm = vectorTwoNorm(vectorSubtraction(x_k, x_star)) / vectorTwoNorm(x_star);
        csvFile << k << "," << r_kTwoNorm << "," << x_min_xstar_twoNorm << "\n";
        std::cout << "Iteration " << k << ": Residual Norm = " << r_kTwoNorm << "Relative error: " << x_min_xstar_twoNorm << std::endl;

        if (r_kTwoNorm < 1e-6) {
            std::cout << "Converged after " << k << " iterations." << std::endl;
            break;
        }
    }


    return x_k;
}

std::vector<double> GSsteepestDescent(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& C, const std::vector<std::vector<double>>& CT, const std::vector<double>& b) {
    std::vector<double> x_star = { 1, 2, 3, 4 };
    int n = A.size();
    //std::vector<double> x = { 1, 2, 3, 4 };
    std::vector<double> x_k(n, 0.0); //x_0 all 0s
    // r_0 = b - Ax_0.. implies r0 = b
    std::vector<double> r_k = b;
    //Solve Pz_0 = r_0
    std::vector<double> y = forwardSubstitution(C, r_k);
    std::vector<double> z_k = backwardSubstitution(CT, y);
    //std::vector<double> z_k = forwardSubstitution(C, r_k);

    std::ofstream csvFile("fcm_data3.csv");
    csvFile << "Iteration,ResidualNorm,RelativeError\n";

    for (int k = 1; k <= 10000; k++) {
        std::vector<double> omega_k = matrixVectorMultiply(A, z_k);
        double alpha_k = dotProduct(z_k, r_k) / dotProduct(z_k, omega_k);
        x_k = vectorAddition(x_k, scalarVectorMultiplication(alpha_k, z_k));
        r_k = vectorSubtraction(r_k, scalarVectorMultiplication(alpha_k, omega_k));
        y = forwardSubstitution(C, r_k);
        z_k = backwardSubstitution(CT, y);

        double r_kTwoNorm = vectorTwoNorm(r_k);
        double x_min_xstar_twoNorm = vectorTwoNorm(vectorSubtraction(x_k, x_star)) / vectorTwoNorm(x_star);
        csvFile << k << "," << r_kTwoNorm << "," << x_min_xstar_twoNorm << "\n";
        std::cout << "Iteration " << k << ": Residual Norm = " << r_kTwoNorm << "Relative error: " << x_min_xstar_twoNorm << std::endl;
   

        if (r_kTwoNorm < 1e-6) {
            std::cout << "Converged after " << k << " iterations." << std::endl;
            break;
        }
    }
    for (int i = 0; i < n; i++) {
        std::cout << x_k[i] << " ";
    }
    csvFile.close();


    return x_k;
}


// -------------------  Conjugate Gradient Methods ------------------------

std::vector<double> IconjugateGradient(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& P, const std::vector<double>& b) {
    std::vector<double> x_star = { 1, 2, 3, 4 };
    int n = A.size();
    std::vector<double> x_k(n, 0.0); //x_0 all 0s
    // r_0 = b - Ax_0.. implies r0 = b
    std::vector<double> r_k = b;
    //Solve Pz_0 = r_0
    std::vector<double> z_k = r_k;
    //Solve p_k = z_k
    std::vector<double> p_k = z_k;

    std::vector<double> x_k_plus_one;
    std::vector<double> r_k_plus_one;
    std::vector<double> z_k_plus_one;
    std::vector<double> p_k_plus_one;

    std::ofstream csvFile("fcm_data4.csv");
    csvFile << "Iteration,ResidualNorm,RelativeError\n";

    for (int k = 1; k <= 10000; k++) {
        std::vector<double> v_k = matrixVectorMultiply(A, p_k);
        double alpha_k = dotProduct(r_k, z_k) / dotProduct(p_k, v_k);
        //print out z_k_plus_one and z_k
        x_k_plus_one = vectorAddition(x_k, scalarVectorMultiplication(alpha_k, p_k));
        r_k_plus_one = vectorSubtraction(r_k, scalarVectorMultiplication(alpha_k, v_k));
        z_k_plus_one = r_k_plus_one;
        double beta_k = dotProduct(r_k_plus_one, z_k_plus_one) / dotProduct(r_k, z_k);
        p_k_plus_one = vectorAddition(z_k_plus_one, scalarVectorMultiplication(beta_k, p_k));

        x_k = x_k_plus_one;
        r_k = r_k_plus_one;
        z_k = z_k_plus_one;
        p_k = p_k_plus_one;

        double r_kTwoNorm = vectorTwoNorm(r_k);
        double x_min_xstar_twoNorm = vectorTwoNorm(vectorSubtraction(x_k, x_star)) / vectorTwoNorm(x_star);
        csvFile << k << "," << r_kTwoNorm << "," << x_min_xstar_twoNorm << "\n";
        std::cout << "Iteration " << k << ": Residual Norm = " << r_kTwoNorm << "Relative error: " << x_min_xstar_twoNorm << std::endl;

        if (r_kTwoNorm < 1e-6) {
            std::cout << "Converged after " << k << " iterations." << std::endl;
            break;
        }

    }
    std::cout << "Solution x: " << std::endl;
    for (int i = 0; i < n; i++) {
        std::cout << x_k[i] << std::endl;
    }

    return x_k;
}

std::vector<double> JconjugateGradient(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& P, const std::vector<double>& b){
    std::vector<double> x_star = { 1, 2, 3, 4 };
    int n = A.size();
    std::vector<double> x_k(n, 0.0); //x_0 all 0s
    // r_0 = b - Ax_0.. implies r0 = b
    std::vector<double> r_k = b;
    //Solve Pz_0 = r_0
    std::vector<double> z_k = J_find_zk(P, b);
    //Solve p_k = z_k
    std::vector<double> p_k = z_k;

    std::vector<double> x_k_plus_one;
    std::vector<double> r_k_plus_one;
    std::vector<double> z_k_plus_one;
    std::vector<double> p_k_plus_one;

    std::ofstream csvFile("fcm_data5.csv");
    csvFile << "Iteration,ResidualNorm,RelativeError\n";

    for (int k = 1; k <= 10000; k++) {
        std::vector<double> v_k = matrixVectorMultiply(A, p_k);
        double alpha_k = dotProduct(r_k, z_k) / dotProduct(p_k, v_k);
        //print out z_k_plus_one and z_k
        x_k_plus_one = vectorAddition(x_k, scalarVectorMultiplication(alpha_k, p_k));
        r_k_plus_one = vectorSubtraction(r_k, scalarVectorMultiplication(alpha_k, v_k));
        z_k_plus_one = J_find_zk(P, r_k_plus_one);
        double beta_k = dotProduct(r_k_plus_one, z_k_plus_one) / dotProduct(r_k, z_k);
        p_k_plus_one = vectorAddition(z_k_plus_one, scalarVectorMultiplication(beta_k, p_k));

        x_k = x_k_plus_one;
        r_k = r_k_plus_one;
        z_k = z_k_plus_one;
        p_k = p_k_plus_one;

        //double r_kTwoNorm = vectorTwoNorm(r_k);
        //std::cout << "Iteration " << k << ": Residual Norm = " << r_kTwoNorm << std::endl;
        //csvFile << k << "," << r_kTwoNorm << "\n";

        double r_kTwoNorm = vectorTwoNorm(r_k);
        double x_min_xstar_twoNorm = vectorTwoNorm(vectorSubtraction(x_k, x_star)) / vectorTwoNorm(x_star);
        csvFile << k << "," << r_kTwoNorm << "," << x_min_xstar_twoNorm << "\n";
        std::cout << "Iteration " << k << ": Residual Norm = " << r_kTwoNorm << "Relative error: " << x_min_xstar_twoNorm << std::endl;

        if (r_kTwoNorm < 1e-6) {
            std::cout << "Converged after " << k << " iterations." << std::endl;
            break;
        }

    }
    std::cout << "Solution x: " << std::endl;
    for (int i = 0; i < n; i++) {
        std::cout << x_k[i] << std::endl;
    }

    return x_k;
}

std::vector<double> GSconjugateGradient(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& C, const std::vector<std::vector<double>>& CT, const std::vector<double>& b) {
    std::vector<double> x_star = { 1, 2, 3, 4 };
    int n = A.size();
    std::vector<double> x_k(n, 0.0); //x_0 all 0s
    // r_0 = b - Ax_0.. implies r0 = b
    std::vector<double> r_k = b;
    //Solve Pz_0 = r_0
    std::vector<double> y = forwardSubstitution(C, r_k);
    std::vector<double> z_k = backwardSubstitution(CT, y);

    //Solve p_k = z_k
    std::vector<double> p_k = z_k;

    std::vector<double> x_k_plus_one;
    std::vector<double> r_k_plus_one;
    std::vector<double> z_k_plus_one;
    std::vector<double> p_k_plus_one;

    std::ofstream csvFile("fcm_data6.csv");
    csvFile << "Iteration,ResidualNorm,RelativeError\n";

    for (int k = 1; k <= 10000; k++) {
        std::vector<double> v_k = matrixVectorMultiply(A, p_k);
        double alpha_k = dotProduct(r_k, z_k) / dotProduct(p_k, v_k);
        //print out z_k_plus_one and z_k
        x_k_plus_one = vectorAddition(x_k, scalarVectorMultiplication(alpha_k, p_k));
        r_k_plus_one = vectorSubtraction(r_k, scalarVectorMultiplication(alpha_k, v_k));

        std::vector<double> y = forwardSubstitution(C, r_k_plus_one);
        std::vector<double> z_k_plus_one = backwardSubstitution(CT, y);

        double beta_k = dotProduct(r_k_plus_one, z_k_plus_one) / dotProduct(r_k, z_k);
        p_k_plus_one = vectorAddition(z_k_plus_one, scalarVectorMultiplication(beta_k, p_k));

        x_k = x_k_plus_one;
        r_k = r_k_plus_one;
        z_k = z_k_plus_one;
        p_k = p_k_plus_one;

        /*double r_kTwoNorm = vectorTwoNorm(r_k);
        std::cout << "Iteration " << k << ": Residual Norm = " << r_kTwoNorm << std::endl;
        csvFile << k << "," << r_kTwoNorm << "\n";*/

        double r_kTwoNorm = vectorTwoNorm(r_k);
        double x_min_xstar_twoNorm = vectorTwoNorm(vectorSubtraction(x_k, x_star)) / vectorTwoNorm(x_star);
        csvFile << k << "," << r_kTwoNorm << "," << x_min_xstar_twoNorm << "\n";
        std::cout << "Iteration " << k << ": Residual Norm = " << r_kTwoNorm << "Relative error: " << x_min_xstar_twoNorm << std::endl;

        if (r_kTwoNorm < 1e-6) {
            std::cout << "Converged after " << k << " iterations." << std::endl;
            break;
        }

    }
    std::cout << "Solution x: " << std::endl;
    for (int i = 0; i < n; i++) {
        std::cout << x_k[i] << std::endl;
    }

    return x_k;

}



int main()
{
    srand(time(0));
    int n;
    std::cout << "Enter the matrix size (n): ";
    std::cin >> n;

  //  std::vector<std::vector<double>> A = alternativeSPD(n);
  // std::vector<std::vector<double>> A = generateSPD(n);
   std::vector <std::vector<double>> P;
   std::vector <std::vector<double>> C;
   std::vector <std::vector<double>> CT;
  // std::vector<double> b = generateVectorB(n);
    
   
   //--------------correctness test---------------------------
   std::vector<std::vector<double>> A = { {5, 7, 6, 5}, {7, 10, 8, 7}, {6, 8, 10, 9}, {5, 7, 9, 10} };
   std::vector<double> b = {57, 79, 88, 86};
   //----------------------------------------------------------

   char method;
   std::cout << "Enter the method of choice(S for steepest descent and C for conjugate gradient): ";
   std::cin >> method;

   char preconditioner;
   std::cout << "Enter the preconditioner choice (I for Identity, J for Jacobi, G for Gauss-Seidel): ";
   std::cin >> preconditioner;

   if (method == 'S' || method == 's') {
       if (preconditioner == 'I' || preconditioner == 'i') {
           P = generateIdentityPreconditioner(n);
           std::vector<double> result = IsteepestDescent(A, P, b);
       }
       else if (preconditioner == 'J' || preconditioner == 'j') {
           P = generateJacobiPreconditioner(A);
           std::vector<double> result = JsteepestDescent(A, P, b);
       }
       else if (preconditioner == 'G' || preconditioner == 'g') {
           C = generateC(A);
           CT = generateCT(A);
           // P = generateGSPreconditioner(A);
           std::vector<double> result = GSsteepestDescent(A, C, CT, b);  
       }
       else {
           std::cout << "Invalid preconditioner choice." << std::endl;
           return 1;
       }

   }
   else if (method == 'C' || method == 'c') {
       if (preconditioner == 'I' || preconditioner == 'i') {
           P = generateIdentityPreconditioner(n);
           std::vector<double> result = IconjugateGradient(A, P, b);
       }
       else if (preconditioner == 'J' || preconditioner == 'j') {
           P = generateJacobiPreconditioner(A);
           std::vector<double> result = JconjugateGradient(A, P, b);
       }
       else if (preconditioner == 'G' || preconditioner == 'g') {
           C = generateC(A);
           CT = generateCT(A);
           // P = generateGSPreconditioner(A);
           std::vector<double> result = GSconjugateGradient(A, C, CT, b);
       }
       else {
           std::cout << "Invalid preconditioner choice." << std::endl;
           return 1;
       }
   }

    

    /*std::cout << "Matrix A:" << std::endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << A[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "vector b: " << std::endl;
    for (int i = 0; i < n; i++) {
        std::cout << b[i] << std::endl;
    }*/


 /*   std::vector <std::vector<double>> CCT = matrixMatrixMultiplication(C, CT);
    std::cout << "Matrix P:" << std::endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << CCT[i][j] << " ";
        }
        std::cout << std::endl;
    }*/
    

    return 0;

}

