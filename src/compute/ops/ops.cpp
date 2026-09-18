module;

#include <expected>
#include <span>

#include <mlt/macros.hpp>
#include <mlt/internal/compute/core/DType.hpp>

#include <cblas.h>

module mlt.internal.compute.operations;

import mlt.internal.compute.core.sizearray;
import mlt.internal.compute.core.mltarray;

using namespace mlt::core;
using namespace mlt::compute::ops;
using namespace mlt::compute::core;

Result<MltArray> mlt::compute::ops::matmul(const MltArray& a, const MltArray& b) noexcept
{
    if (a.dType != b.dType)
        return std::unexpected(MltError::makeTypeMismatch(toString(a.dType), toString(b.dType)));

    // if (a.shape.size() < 2 || b.shape.size() < 2)
    //     return std::unexpected(MltError::makeShapeMismatch());

    // temporary matrix check
    if (a.shape.size() != 2 || b.shape.size() != 2)
        return std::unexpected(MltError::makeShapeMismatch());

    // Does only work as long the above check is here
    bool aTransposed = a.isTransposed();
    bool bTransposed = a.isTransposed();
    
    constexpr size_t COL = 1;
    constexpr size_t ROW = 0;
    
    std::span<const size_t> aShape = a.shape.asSpan().last(2);
    std::span<const size_t> bShape = b.shape.asSpan().last(2);
    
    if (aShape[COL] != bShape[ROW] || aShape[ROW] != bShape[COL])
        return std::unexpected(MltError::makeShapeMismatch());

    const size_t M = aShape[ROW];
    const size_t K = aShape[COL];
    const size_t N = bShape[COL];

    DefaultSizeArray resultShape = DefaultSizeArray({M, N});
    MLT_TRY(result, MltArray::from(resultShape));
    
    const float* A = a.asPtrUnchecked<float>();
    const float* B = b.asPtrUnchecked<float>();
    float* C = result.asPtrUnchecked<float>();

    cblas_sgemm(
        CblasRowMajor,
        aTransposed ? CblasTrans : CblasNoTrans,
        bTransposed ? CblasTrans : CblasNoTrans,
        static_cast<int>(M),
        static_cast<int>(N),
        static_cast<int>(K),
        1.0f,
        A,
        static_cast<int>(K),
        B,
        static_cast<int>(N),
        0.0f,
        C,
        static_cast<int>(N)
    );

    return result;
}

Result<MltArray> mlt::compute::ops::add(const MltArray& a, const MltArray& b) noexcept
{
    if (a.dType != b.dType)
        return std::unexpected(MltError::makeTypeMismatch(toString(a.dType), toString(b.dType)));

    if (a.shape.size() != b.shape.size())
        return std::unexpected(MltError::makeShapeMismatch());
    
    for (size_t i = 0; i < a.shape.size(); ++i)
        if (a.shape[i] != b.shape[i])
            return std::unexpected(MltError::makeShapeMismatch());

    return MltArray::from(DefaultSizeArray({2}));
}

