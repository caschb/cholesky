#include <iostream>

#include <Eigen/Cholesky>
#include <Eigen/Dense>

#define CUT 5

using namespace Eigen;

using MatType = MatrixXd;

template <typename Derived>
void triangular_solve(const MatrixBase<Derived> &A,
                      const MatrixBase<Derived> &B,
                      MatrixBase<Derived> &result) {
  result = A * B.transpose().inverse();
}

template <typename Derived>
void symmetric_rank_update(MatrixBase<Derived> &A, MatrixBase<Derived> &C) {

  A = A - C * C.transpose();
}

template <typename Derived>
void cholesky_decomp(MatrixBase<Derived> &A, MatrixBase<Derived> &solution) {
  int dim = A.cols();
  if (dim * dim < CUT) {
    LLT<MatType> lltcompute(A);
    solution = lltcompute.matrixL();
  } else {
    MatType L00;
    MatType A00 = A.topLeftCorner(dim / 2, dim / 2);
    cholesky_decomp(A00, L00);
    MatType A10 = A.bottomLeftCorner(dim / 2, dim / 2);
    MatType L10;
    triangular_solve(A10, L00, L10);
    MatType A11 = A.bottomRightCorner(dim / 2, dim / 2);
    MatType L11;
    symmetric_rank_update(A11, L10);
    cholesky_decomp(A11, L11);
    solution = MatType(dim, dim);
    solution.setZero();
    solution.topLeftCorner(A00.cols(), A00.rows()) = L00;
    solution.bottomLeftCorner(A10.cols(), A10.rows()) = L10;
    solution.bottomRightCorner(A11.cols(), A11.rows()) = L11;
  }
}

int main(int argc, char **argv) {
  int dimensions = 3;
  if (argc > 1) {
    dimensions = std::atoi(argv[1]);
  }

  MatType q = MatType::Random(dimensions, dimensions);
  MatType diag(dimensions, dimensions);

  for (int i = 0; i < dimensions; ++i) {
    diag(i, i) = i + 1;
  }

  MatType posdefinite = q.transpose() * diag * q;

  MatType lower;
  cholesky_decomp(posdefinite, lower);

  LLT<MatType> lltcompute(posdefinite);
  MatType compare = lltcompute.matrixL();

  std::cout << compare.isApprox(lower) << '\n';

  return 0;
}
