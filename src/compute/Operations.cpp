module;

#include <expected>
#include <span>

#include <mlt/macros.hpp>
#include <mlt/internal/compute/core/DType.hpp>

#include <cblas.h>

module mlt.internal.compute.operations;

import mlt.internal.compute.core.sizearray;
import mlt.internal.compute.core.mltarray;
import mlt.internal.compute.runtime;

using namespace mlt::core;
using namespace mlt::compute::ops;
using namespace mlt::compute::core;

Result<MltArray> multiplyMatrixVectorKernel(const MltArray& matrix, const MltArray& vector)
{
    // cblas_sgemv(const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE TransA, const int32_t M, const int32_t N, const float alpha, const float *A, const int32_t lda, const float *X, const int32_t incX, const float beta, float *Y, const int32_t incY);
    // cblas_dgemv(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA, const int32_t M, const int32_t N, const double alpha, const double *A, const int32_t lda, const double *X, const int32_t incX, const double beta, double *Y, const int32_t incY);
}

Result<MltArray> mlt::compute::ops::multiply(const MltArray& a, const MltArray& b, const size_t dim) noexcept
{
    if (a.dType != b.dType)
        return std::unexpected(MltError::makeTypeMismatch(toString(a.dType), toString(b.dType)));

    return mlt::compute::runtime::multiply(a, b, dim);
}

Result<MltArray> mlt::compute::ops::matmul(const MltArray& a, const MltArray& b) noexcept
{
    if (a.dType != b.dType)
        return std::unexpected(MltError::makeTypeMismatch(toString(a.dType), toString(b.dType)));

    // if (a.shape.size() < 2 || b.shape.size() < 2)
    //     return std::unexpected(MltError::makeShapeMismatch());

    // temporary matrix check
    if (!a.isMatrix() || !b.isMatrix())
        return std::unexpected(MltError::makeShapeMismatch());

    // Does only work as long the above check is here
    bool aTransposed = a.isTransposed();
    bool bTransposed = b.isTransposed();
    
    constexpr size_t COL = 1;
    constexpr size_t ROW = 0;
    
    std::span<const size_t> aShape = a.shape.asSpan().last(2);
    std::span<const size_t> bShape = b.shape.asSpan().last(2);
    
    if (aShape[COL] != bShape[ROW])
        return std::unexpected(MltError::makeShapeMismatch());

    return mlt::compute::runtime::matmul(a, b);
}

Result<MltArray> mlt::compute::ops::add(const MltArray& a, const MltArray& b) noexcept
{
    if (a.dType != b.dType)
        return std::unexpected(MltError::makeTypeMismatch(toString(a.dType), toString(b.dType)));

    if (a.shape.size() != b.shape.size())
        return std::unexpected(MltError::makeShapeMismatch());

    if (!a.shape.equals(b.shape))
        return std::unexpected(MltError::makeShapeMismatch());


    return MltArray::from(DefaultSizeArray({2}));
}

