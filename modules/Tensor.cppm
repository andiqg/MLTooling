module;

#include <mlt/macros.hpp>

#include <cstddef>
#include <span>
#include <vector>

export module mlt.compute.Tensor;

import mlt.core.error;
import mlt.internal.compute.core.mltarray;

namespace mlt::compute
{
    export class Tensor
    {
        mlt::compute::core::MltArray mArray;
        
        Tensor(mlt::compute::core::MltArray array) noexcept : mArray(std::move(array))
        {}

        public:
        Tensor(const Tensor&) noexcept = default;
        Tensor& operator=(const Tensor&) noexcept = default;
        Tensor(Tensor&&) noexcept = default;
        Tensor& operator=(Tensor&&) noexcept = default;

        static mlt::core::Result<Tensor> from(std::initializer_list<size_t> shape) noexcept;
        static mlt::core::Result<Tensor> from(std::span<const float> buf, std::initializer_list<size_t> shape) noexcept;
        static mlt::core::Result<Tensor> from(std::span<const size_t> shape) noexcept;
        static mlt::core::Result<Tensor> from(std::span<const float> buf, std::span<const size_t> shape) noexcept;

        Tensor transpose() const noexcept;

        template <typename... Indices>
        mlt::core::Result<float*> at(Indices... idx) noexcept
        {
            MLT_TRY(val, mArray.at(idx...));
            return val;
        }

        template <typename... Indices>
        mlt::core::Result<float> at(Indices... idx) const noexcept
        {
            MLT_TRY(val, mArray.at(idx...));
            return val;
        }

        mlt::core::Result<float*> at(std::span<const size_t> indices) noexcept;
        mlt::core::Result<float> at(std::span<const size_t> indices) const noexcept;
        std::vector<size_t> shape() const;
        mlt::core::Result<Tensor> clone() const;

        mlt::core::Result<Tensor> matmul(const Tensor& other) const noexcept;
        mlt::core::Result<Tensor> multiply(const Tensor& other) const noexcept;

        mlt::core::Result<Tensor> add(const Tensor& other) const noexcept;
        mlt::core::Result<Tensor> sub(const Tensor& other) const noexcept;
    };

    mlt::core::Result<Tensor> matmul(const Tensor& a, const Tensor& b) noexcept;
    mlt::core::Result<Tensor> multiply(const Tensor& a, const Tensor& b) noexcept;

    mlt::core::Result<Tensor> add(const Tensor& a, const Tensor& b) noexcept;
    mlt::core::Result<Tensor> sub(const Tensor& a, const Tensor& b) noexcept;
}
