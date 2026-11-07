module;

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>

#include <mlt/macros.hpp>
#include <mlt/internal/compute/core/DType.hpp>

module mlt.internal.compute.runtime;

import mlt.internal.compute.core.sizearray;
import mlt.internal.compute.kernels;

using namespace mlt::compute::runtime;

using namespace mlt::core;
using namespace mlt::compute::core;

// ===== Private Functions =====

Result<MltArray> dispatchScale(const MltArray& a, const MltArray& scalar) noexcept
{
    constexpr int32_t steps = 1;
    const int32_t N = static_cast<int32_t>(a.shape.elemCount());

    switch (a.dType)
    {
        case DType::FLOAT32:
        {
            float alpha = scalar.asPtrUnchecked<float>()[0];
            MLT_TRY(b, a.clone());
            float* X = b.asPtrUnchecked<float>();

            mlt::compute::kernels::blasScaleF32(N, alpha, X, steps); 

            return b;
        }

        case DType::FLOAT64:
        {
            double alpha = scalar.asPtrUnchecked<double>()[0];
            MLT_TRY(b, a.clone());
            double* X = b.asPtrUnchecked<double>();

            mlt::compute::kernels::blasScaleF64(N, alpha, X, steps);

            return b;
        }
    }
}

template <mlt::compute::kernels::BinaryOp Op>
Result<MltArray> dispatchBinaryBroadcast(const MltArray& a, const MltArray& b, size_t dim) noexcept
{
    const bool aIsHost = b.isScalar() ? true : a.isScalar() ? false : a.shape.size() >= b.shape.size();
    const MltArray& host = aIsHost ? a : b;
    const MltArray& op = aIsHost ? b : a;

    size_t outer = 1;
    size_t mid = 1;
    size_t inner = host.shape.elemCount();

    if (!op.isScalar())
    {
        const size_t end = host.shape.size() - dim;
        const size_t begin = end - op.shape.size();

        inner = 1;

        for (size_t i = 0; i < begin; ++i)
            outer *= host.shape[i];

       for (size_t i = end; i < host.shape.size(); ++i)
            inner *= host.shape[i];

        mid = op.shape.elemCount();
    }
    
    assert(outer * mid * inner == host.shape.elemCount());

    MLT_TRY(out, host.clone());

    switch (a.dType) 
    {
        case DType::FLOAT32:
        {
            mlt::compute::kernels::broadcastInplace<Op>(out.asPtrUnchecked<float>(), op.asPtrUnchecked<float>(), aIsHost, outer, mid, inner);
            return out;
        }
        case DType::FLOAT64:
        {
            mlt::compute::kernels::broadcastInplace<Op>(out.asPtrUnchecked<double>(), op.asPtrUnchecked<double>(), aIsHost, outer, mid, inner);
            return out;
        }
    }
}

// ===== Public Functions =====

Result<MltArray> mlt::compute::runtime::multiply(const MltArray& a, const MltArray& b, size_t dim) noexcept
{
    if (a.isScalar())
        return dispatchScale(b, a);
    
    if (b.isScalar())
        return dispatchScale(a, b);

    return dispatchBinaryBroadcast<mlt::compute::kernels::BinaryOp::Mul>(a, b, dim);
}

Result<MltArray> mlt::compute::runtime::add(const MltArray& a, const MltArray& b, size_t dim) noexcept
{
    return dispatchBinaryBroadcast<mlt::compute::kernels::BinaryOp::Add>(a, b, dim);
}

Result<MltArray> mlt::compute::runtime::sub(const MltArray& a, const MltArray& b, size_t dim) noexcept
{
    return dispatchBinaryBroadcast<mlt::compute::kernels::BinaryOp::Sub>(a, b, dim);
}

Result<MltArray> mlt::compute::runtime::matmul(const MltArray& a, const MltArray& b) noexcept
{
    constexpr size_t COL = 1;
    constexpr size_t ROW = 0;

    std::span<const size_t> aShape = a.shape.asSpan().last(2);
    std::span<const size_t> bShape = b.shape.asSpan().last(2);

    bool aTransposed = a.isTransposed();
    bool bTransposed = b.isTransposed();

    const size_t m = aShape[ROW];
    const size_t k = aShape[COL];
    const size_t n = bShape[COL];

    DefaultSizeArray resultShape = DefaultSizeArray({ m , n });
    MLT_TRY(result, MltArray::from(resultShape, a.dType, a.dimType));

    const int32_t M = static_cast<int32_t>(m);
    const int32_t K = static_cast<int32_t>(k);
    const int32_t N = static_cast<int32_t>(n);

    switch (a.dType) 
    {
        case mlt::compute::core::DType::FLOAT32:
        {
            mlt::compute::kernels::blasMatmulF32(
                M, N, K,
                1.f,
                a.asPtrUnchecked<float>(), K,
                b.asPtrUnchecked<float>(), N,
                0.0f,
                result.asPtrUnchecked<float>(), N,
                false, aTransposed, bTransposed
            );

            break;
        }
    
        case mlt::compute::core::DType::FLOAT64:
        {
            mlt::compute::kernels::blasMatmulF64(
                M, N, K,
                1.f,
                a.asPtrUnchecked<double>(), K,
                b.asPtrUnchecked<double>(), N,
                0.f,
                result.asPtrUnchecked<double>(), N,
                false, aTransposed, bTransposed
            );

            break;
        }
    }

    return result;
}

