#include "mlt/core/error.h"
#include <algorithm>
#include <mlt/compute/tensor.h>


#include <stdio.h>
#include <span>
#include <vector>

import mlt.core.error;

import mlt.compute.Tensor;

#ifdef __cplusplus
extern "C"
{
    using namespace mlt::compute;
    using namespace mlt::core;

    struct MLTtensor
    {
        Tensor implTensor;

        MLTtensor(Tensor tensor) : implTensor(tensor) {}
    };

    constexpr const char* INTERNAL_ERROR = "Internal error.";

    MLTerrorType convertErrorType(mlt::core::MltErrorType type)
    {
        switch(type)
        {
            case MltErrorType::OutOfMemory:      return MLT_OUT_OF_MEMORY;
            case MltErrorType::InvalidAlignment: return MLT_INVALID_ALIGNMENT;
            case MltErrorType::OutOfBounds:      return MLT_OUT_OF_BOUNDS;
            case MltErrorType::TypeMismatch:     return MLT_TYPE_MISMATCH;
            case MltErrorType::RankMismatch:     return MLT_RANK_MISMATCH;
            case MltErrorType::ShapeMismatch:    return MLT_SHAPE_MISMATCH;
            case MltErrorType::InvalidArgument:  return MLT_INVALID_ARGUMENT;
        }
    }

    MLTerror mltTensorFrom(MLTshape shape, MLTtensor** out)
    {
        // nullptr check for out
        if (!out)
            return mltErrorCreate(MLT_NULL_PTR, "out cannot be NULL.");
        
        try
        {
            Result<Tensor> result = Tensor::from(std::span<const size_t>(shape.dims, shape.size));

            if (!result)
                return mltErrorCreate(convertErrorType(result.error().type), result.error().msg.data());

            *out = new MLTtensor(result.value());
            return mltErrorCreate(MLT_ERROR_NONE, "");
        }
        catch(...)
        {
            MLTerror err = mltErrorCreate(MLT_INTERNAL_ERROR, INTERNAL_ERROR);
            return err;
        }
    }

    MLTerror mltTensorFromBuff(const float* buf, size_t len, MLTshape shape, MLTtensor** out)
    {
        if (!out)
            return mltErrorCreate(MLT_NULL_PTR, "out cannot be NULL.");

        if (!buf)
            return mltErrorCreate(MLT_NULL_PTR, "buf cannot be NULL.");
        
        try
        {
            Result<Tensor> result = Tensor::from(std::span<const float>(buf, len), std::span<const size_t>(shape.dims, shape.size));

            if (!result)
                return  mltErrorCreate(convertErrorType(result.error().type), result.error().msg.data());

            *out = new MLTtensor(result.value());
            return mltErrorCreate(MLT_ERROR_NONE, "");
        }
        catch(...)
        {
            MLTerror err = mltErrorCreate(MLT_INTERNAL_ERROR, INTERNAL_ERROR);
            return err;
        }
    }

    void mltTensorDestroy(MLTtensor* tensor)
    {
        if (!tensor)
            return;

        delete tensor;

    }

    MLTerror mltTensorCopy(const MLTtensor *tensor, MLTtensor **out)
    {
        try
        {
            if (!out)
                return mltErrorCreate(MLT_INVALID_ARGUMENT, "out cannot be NULL.");

            *out = new MLTtensor(tensor->implTensor);
            return mltErrorCreate(MLT_ERROR_NONE, "");
        }
        catch(...)
        {
            MLTerror err = mltErrorCreate(MLT_INTERNAL_ERROR, INTERNAL_ERROR);
            return err;
        }
    }

    MLTerror mltTensorShape(const MLTtensor *tensor, MLTshape *shape)
    {
        try
        {
            if (!tensor)
                return mltErrorCreate(MLT_NULL_PTR, "tensor cannot be NULL.");

            if (!shape)
                return mltErrorCreate(MLT_NULL_PTR, "shape cannot be NULL.");

            std::vector<size_t> shapeVec = tensor->implTensor.shape();

            if (shapeVec.size() > MLT_MAX_DIMS)
                return mltErrorCreate(MLT_INTERNAL_ERROR, "shape problem.");

            MLTshape shape{};
            shape.size = shapeVec.size();
            std::copy(shapeVec.begin(), shapeVec.end(), shape.dims);
            

            return mltErrorCreate(MLT_ERROR_NONE, "");
        }
        catch(...)
        {
            return mltErrorCreate(MLT_INTERNAL_ERROR, INTERNAL_ERROR);
        }
    }

    MLTerror mltTensorGet(const MLTtensor *tensor, MLTshape shape, float *out)
    {
        try
        {
            if (!out)
                return mltErrorCreate(MLT_NULL_PTR, "out cannot be NULL.");

            if (!tensor)
                return mltErrorCreate(MLT_NULL_PTR, "tensor cannot be NULL.");

            Result<float> res = tensor->implTensor.at(std::span<const size_t>(shape.dims, shape.size));

            if (!res)
                return mltErrorCreate(convertErrorType(res.error().type), res.error().msg.data());

            *out = res.value();

            return mltErrorCreate(MLT_ERROR_NONE, "");
        }
        catch (...)
        {
            MLTerror err = mltErrorCreate(MLT_INTERNAL_ERROR, INTERNAL_ERROR);
            return err;
        }
    }

    MLTerror mltTensorSet(MLTtensor *tensor, MLTshape shape, float val)
    {
        try 
        {
            if (!tensor)
                return mltErrorCreate(MLT_NULL_PTR, "tensor cannot be NULL.");

            Result<float*> res = tensor->implTensor.at(std::span<const size_t>(shape.dims, shape.size));
            
            if (!res)
                return mltErrorCreate(convertErrorType(res.error().type), res.error().msg.data());

            float* value = res.value();
            *value = val;

            return mltErrorCreate(MLT_ERROR_NONE, "");
        }
        catch (...)
        {
            return mltErrorCreate(MLT_INTERNAL_ERROR, INTERNAL_ERROR);
        }
    }

    MLTerror mltTensorMatmul(const MLTtensor *a, const MLTtensor *b, MLTtensor **out)
    {
        try 
        {
            if (!a)
                return mltErrorCreate(MLT_NULL_PTR, "a cannot be NULL.");
        
            if (!b)
                return mltErrorCreate(MLT_NULL_PTR, "b cannot be NULL.");

            Result<Tensor> res = a->implTensor.matmul(b->implTensor);

            if (!res)
                return mltErrorCreate(convertErrorType(res.error().type), res.error().msg.data());

            *out = new MLTtensor(res.value());
            return mltErrorCreate(MLT_ERROR_NONE, "");
        }
        catch (...)
        {
            return mltErrorCreate(MLT_INTERNAL_ERROR, INTERNAL_ERROR);
        }
    }
}
#endif
