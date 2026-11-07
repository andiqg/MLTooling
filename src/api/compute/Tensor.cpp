module;

#include <mlt/macros.hpp>

#include <cstddef>
#include <span>
#include <utility>
#include <vector>

module mlt.compute.Tensor;
import mlt.internal.compute.core.mltarray;
import mlt.internal.compute.core.sizearray;
import mlt.internal.compute.operations;

using namespace mlt::core;
using namespace mlt::compute;
using namespace mlt::compute::core;

Result<Tensor> Tensor::from(std::initializer_list<size_t> shape) noexcept
{
    MLT_TRY(arr, MltArray::from(DefaultSizeArray(shape)));
    return Tensor(std::move(arr));
}

Result<Tensor> Tensor::from(std::span<const float> buf, std::initializer_list<size_t> shape) noexcept
{
    MLT_TRY(arr, MltArray::from(buf, DefaultSizeArray(shape)));
    return Tensor(std::move(arr));
}

Result<Tensor> Tensor::from(std::span<const size_t> shape) noexcept
{
    MLT_TRY(arr, MltArray::from(DefaultSizeArray(shape)));
    return Tensor(std::move(arr));
}

Result<Tensor> Tensor::from(std::span<const float> buf, std::span<const size_t> shape) noexcept
{
    MLT_TRY(arr, MltArray::from(buf, DefaultSizeArray(shape)));
    return Tensor(std::move(arr));
}

Tensor Tensor::transpose() const noexcept
{
    MltArray transposedArr = mArray.transpose();
    return Tensor(transposedArr);
}

Result<float*> Tensor::at(std::span<const size_t> indices) noexcept
{
    MLT_TRY(val, mArray.at(indices));
    return val;
}

Result<float> Tensor::at(std::span<const size_t> indices) const noexcept
{
    MLT_TRY(val, mArray.at(indices));
    return val;
}

std::vector<size_t> Tensor::shape() const
{
    std::span<const size_t> shapeSpan =  mArray.shape.asSpan();
    return std::vector(shapeSpan.begin(), shapeSpan.end());
}

Result<Tensor> Tensor::matmul(const Tensor& other) const noexcept
{
    MLT_TRY(arr, mlt::compute::ops::matmul(mArray, other.mArray));
    return Tensor(std::move(arr));
}

Result<Tensor> Tensor::multiply(const Tensor& other) const noexcept
{
    MLT_TRY(arr, mlt::compute::ops::multiply(mArray, other.mArray, 0));
    return Tensor(std::move(arr));
}

// ==== Free functions ===
Result<Tensor> matmul(const Tensor& a, const Tensor& b) noexcept
{
    return a.matmul(b);
}

Result<Tensor> multiply(const Tensor& a, const Tensor& b) noexcept
{
    return a.multiply(b);
}
