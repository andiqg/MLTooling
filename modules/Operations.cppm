module;

#include <cstddef>

export module mlt.internal.compute.operations;

import mlt.core.error;
import mlt.internal.compute.core.mltarray;

export namespace mlt::compute::ops
{
    mlt::core::Result<core::MltArray> add(const core::MltArray& a, const core::MltArray& b, size_t dim) noexcept;
    mlt::core::Result<core::MltArray> sub(const core::MltArray& a, const core::MltArray& b, size_t dim) noexcept;
    mlt::core::Result<core::MltArray> multiply(const core::MltArray& a, const core::MltArray& b, size_t dim) noexcept;

    mlt::core::Result<core::MltArray> matmul(const core::MltArray& a, const core::MltArray& b) noexcept;
}
