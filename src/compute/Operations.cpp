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

inline Result<void> validateBinaryOperations(const MltArray& a, const MltArray& b, size_t dim) noexcept
{
    if (a.isScalar() || b.isScalar())
        return {};

    const bool aIsHost = b.isScalar() ? true : a.isScalar() ? false : a.shape.size() >= b.shape.size();
    const MltArray& host = aIsHost ? a : b;
    const MltArray& op = aIsHost ? b : a;

    if (dim + op.shape.size() > host.shape.size())
        return std::unexpected(MltError::makeInvalidBroadcastDim());

    const size_t begin = host.shape.size() - dim - op.shape.size();

    for (size_t j = 0; j < op.shape.size(); ++j)
    {
        if (op.shape[j] != host.shape[begin + j])
            return std::unexpected(MltError::makeShapeMismatch());
    }

    return {};
}

inline Result<void> validateType(const MltArray& a, const MltArray& b) noexcept
{
    if (a.dType != b.dType)
        return std::unexpected(MltError::makeTypeMismatch(toString(a.dType), toString(b.dType)));

    return {};
}

Result<MltArray> mlt::compute::ops::multiply(const MltArray& a, const MltArray& b, const size_t dim) noexcept
{
    MLT_RETURN_IF_ERROR(validateType(a, b));
    MLT_RETURN_IF_ERROR(validateBinaryOperations(a, b, dim));

    return mlt::compute::runtime::multiply(a, b, dim);
}

Result<MltArray> mlt::compute::ops::matmul(const MltArray& a, const MltArray& b) noexcept
{
    MLT_RETURN_IF_ERROR(validateType(a, b));

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

Result<MltArray> mlt::compute::ops::add(const MltArray& a, const MltArray& b, size_t dim) noexcept
{
    MLT_RETURN_IF_ERROR(validateType(a, b));
    MLT_RETURN_IF_ERROR(validateBinaryOperations(a, b, dim));

    return mlt::compute::runtime::add(a, b, dim);
}

Result<MltArray> mlt::compute::ops::sub(const MltArray& a, const MltArray& b, size_t dim) noexcept
{
    MLT_RETURN_IF_ERROR(validateType(a, b));
    MLT_RETURN_IF_ERROR(validateBinaryOperations(a, b, dim));

    return mlt::compute::runtime::sub(a, b, dim);
}

