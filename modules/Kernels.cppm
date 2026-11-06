module;

#include <cstdint>
#include <cstdlib>

export module mlt.internal.compute.kernels;

export namespace mlt::compute::kernels
{
    // void nativeScaleF32(int32_t N, float alpha, float* X, int32_t incX) noexcept;
    // void nativeScaleF64(int32_t N, double alpha, double* X, int32_t incX) noexcept;

    void blasScaleF32(int32_t N, float alpha, float* X, int32_t incX) noexcept;
    void blasScaleF64(int32_t N, double alpha, double* X, int32_t incX) noexcept;

    void blasMatmulF32(
        int32_t M,
        int32_t N,
        int32_t K,
        float alpha,
        const float* A, 
        int32_t lda,
        const float* B,
        int32_t ldb,
        float beta,
        float* C,
        int32_t ldc,
        bool colMajor,
        bool aTransposed,
        bool bTransposed
    ) noexcept;

     void blasMatmulF64(
        int32_t M,
        int32_t N,
        int32_t K,
        double alpha,
        const double* A, 
        int32_t lda,
        const double* B,
        int32_t ldb,
        double beta,
        double* C,
        int32_t ldc,
        bool colMajor,
        bool aTransposed,
        bool bTransposed
    ) noexcept;
    
    template <typename T>
    void hardamardAligned(T* x, const T* p, size_t outer, size_t mid, size_t inner)
    {
        for (size_t o = 0; o < outer; ++o)
        {
            T* block = x + o * mid * inner;

            for (size_t m = 0; m < mid; ++m)
            {
                const T s = p[m];
                T* row = block + m * inner;

                for (size_t i = 0; i < inner; ++i)
                    row[i] *= s;
            }
        }
    }
}

