module;

#include "mlt/macros.hpp"
#include <expected>

#include <mlt/internal/compute/core/RefCount.hpp>
#include <mlt/internal/compute/core/DType.hpp>

module mlt.internal.compute.core.mltarray;

import mlt.core.error;
import mlt.internal.core.storage;
import mlt.internal.compute.core.sizearray;

using namespace mlt::core;
using namespace mlt::compute::core;

MltArray::MltArray(
    Ref<Storage> storage,
    DefaultSizeArray shape,
    DefaultSizeArray strides,
    DType dType,
    DimType dimType
) noexcept
    :   data(storage),
        shape(shape),
        strides(strides),
        offset(0),
        dType(dType),
        dimType(dimType)
{}

Result<MltArray> MltArray::from(DefaultSizeArray shape, const DType dType, const DimType dimType) noexcept
{
    const size_t bytes = shape.elemCount();
    
    if (bytes == 0)
        return std::unexpected(mlt::core::MltError::makeInvalidArgument());

    DefaultSizeArray strides = DefaultSizeArray(shape.size());
    computeStrides(strides, shape, 1, dimType);
    
    mlt::core::Result<Ref<Storage>> cStorage = Storage::alloc(bytes * toByteCount(dType));

    if (!cStorage)
        return std::unexpected(cStorage.error());
    
    Ref<Storage> storage = cStorage.value();

    return MltArray(
        storage,
        shape,
        strides,
        dType,
        dimType
    );
}

MltArray MltArray::transpose() const noexcept
{
    MltArray transposed = *this;
    const size_t shapeLen = transposed.shape.size();

    for (size_t i = 0; i < shapeLen / 2; ++i)
    {
        const size_t j = shapeLen - 1 - i;

        std::swap(transposed.shape[i], transposed.shape[j]);
        std::swap(transposed.strides[i], transposed.strides[j]);
    }

    return transposed;
}

bool MltArray::isContiguous() const
{
    if (shape.size() == 0)
        return true;
    
    size_t expected = 1;

    for (size_t i = shape.size(); i-- > 0;)
    {
        if (strides[i] != expected)
            return false;
        
        expected *= shape[i];
    }

    return true;
}

Result<MltArray> MltArray::clone() const noexcept
{
    MLT_TRY(cloned, data->clone());

    MltArray arr = MltArray(cloned, shape, strides, dType, dimType);
    arr.offset = offset;

    return arr;
}
