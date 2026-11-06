module;

#include <cblas.h>
#include <cstdint>

module mlt.internal.compute.kernels;

import mlt.core.error;

using namespace mlt::core;
using namespace mlt::compute::kernels;

void mlt::compute::kernels::blasScaleF32(const int32_t N, const float alpha, float* X, const int32_t incX) noexcept
{
    cblas_sscal(N, alpha, X,  incX);
}

void mlt::compute::kernels::blasScaleF64(const int32_t N, const double alpha, double* X, const int32_t incX) noexcept
{
    cblas_dscal(N, alpha, X, incX);
}

void mlt::compute::kernels::blasMatmulF32(
        const int32_t M,
        const int32_t N,
        const int32_t K,
        const float alpha,
        const float* A, 
        const int32_t lda,
        const float* B,
        const int32_t ldb,
        const float beta,
        float* C,
        const int32_t ldc,
        const bool colMajor,
        const bool aTransposed,
        const bool bTransposed
) noexcept
{
    cblas_sgemm(
        colMajor ? CblasColMajor : CblasRowMajor,
        aTransposed ? CblasTrans : CblasNoTrans,
        bTransposed ? CblasTrans : CblasNoTrans,
        M,
        N,
        K,
        alpha,
        A, 
        lda, 
        B, 
        ldb, 
        beta, 
        C, 
        ldc
    );
}

void mlt::compute::kernels::blasMatmulF64(
    const int32_t M,
    const int32_t N,
    const int32_t K,
    const double alpha,
    const double* A,
    const int32_t lda,
    const double* B,
    const int32_t ldb,
    const double beta,
    double* C,
    const int32_t ldc,
    bool colMajor,
    bool aTransposed,
    bool bTransposed
) noexcept
{
    cblas_dgemm(
        colMajor ? CblasColMajor : CblasRowMajor,
        aTransposed ? CblasTrans : CblasNoTrans,
        bTransposed ? CblasTrans : CblasNoTrans,
        M,
        N,
        K,
        alpha,
        A,
        lda,
        B,
        ldb,
        beta,
        C,
        ldc
    );
}

