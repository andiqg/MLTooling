module;

#include <cstddef>

export module mlt.internal.compute.runtime;

import mlt.core.error;
import mlt.internal.compute.core.mltarray;

import mlt.internal.compute.kernels;

export namespace mlt::compute::runtime 
{
    mlt::core::Result<core::MltArray> add(const core::MltArray& a, const core::MltArray& b, size_t dim) noexcept;
    mlt::core::Result<core::MltArray> sub(const core::MltArray& a, const core::MltArray& b, size_t dim) noexcept;
    mlt::core::Result<core::MltArray> multiply(const core::MltArray& a, const core::MltArray& b, size_t dim) noexcept; 

    mlt::core::Result<core::MltArray> matmul(const core::MltArray& a, const core::MltArray& b) noexcept;
}
