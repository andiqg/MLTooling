#include "mlt/core/error.h"
#include <algorithm>
#include <mlt/compute/tensor.h>
#include <mlt/macros.hpp>


#include <stdio.h>
#include <span>
#include <vector>

import mlt.core.error;

import mlt.compute.Tensor;


#ifdef __cplusplus
extern "C"
{
    #define MLT_CHECK_ERR(result)                                                                           \
        do {                                                                                                \
            if (!result)                                                                                    \
                return mltErrorCreate(convertErrorType(result.error().type), result.error().msg.data());    \
        } while(0)

    using namespace mlt::compute;
    using namespace mlt::core;

    struct MLTtensor
    {
        Tensor implTensor;

        MLTtensor(Tensor tensor) : implTensor(std::move(tensor)) {}
    };

    MLTerrorType convertErrorType(mlt::core::MltErrorType type)
    {
        switch(type)
        {
            case MltErrorType::OutOfMemory:         return MLT_OUT_OF_MEMORY;
            case MltErrorType::InvalidAlignment:    return MLT_INVALID_ALIGNMENT;
            case MltErrorType::OutOfBounds:         return MLT_OUT_OF_BOUNDS;
            case MltErrorType::TypeMismatch:        return MLT_TYPE_MISMATCH;
            case MltErrorType::RankMismatch:        return MLT_RANK_MISMATCH;
            case MltErrorType::ShapeMismatch:       return MLT_SHAPE_MISMATCH;
            case MltErrorType::InvalidBroadcastDim: return MLT_INVALID_BROADCAST_DIM;
            case MltErrorType::InvalidArgument:     return MLT_INVALID_ARGUMENT;
        }
    }

    MLTerror mltTensorFrom(MLTshape shape, MLTtensor** out)
    {
        MLT_CHECK_NULL_PTR(out);
        
        try
        {
            Result<Tensor> result = Tensor::from(std::span<const size_t>(shape.dims, shape.size));
            MLT_CHECK_ERR(result);

            *out = new MLTtensor(MLT_MOVE_VALUE(result));
            return mltMakeErrorNone();
        }
        catch(...)
        {
            return mltMakeErrorInternal();
        }
    }

    MLTerror mltTensorFromBuff(const float* buf, size_t len, MLTshape shape, MLTtensor** out)
    {
        MLT_CHECK_NULL_PTR(out);
        MLT_CHECK_NULL_PTR(buf);

        try
        {
            Result<Tensor> result = Tensor::from(std::span<const float>(buf, len), std::span<const size_t>(shape.dims, shape.size));
            MLT_CHECK_ERR(result);

            *out = new MLTtensor(MLT_MOVE_VALUE(result));
            return mltMakeErrorNone();
        }
        catch(...)
        {
            return mltMakeErrorInternal();
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
        MLT_CHECK_NULL_PTR(tensor);
        MLT_CHECK_NULL_PTR(out);

        try
        {
            *out = new MLTtensor(tensor->implTensor);
            return mltMakeErrorNone();
        }
        catch(...)
        {
            return mltMakeErrorInternal();
        }
    }

    MLTerror mltTensorClone(const MLTtensor *tensor, MLTtensor **out)
    {
        MLT_CHECK_NULL_PTR(tensor);
        MLT_CHECK_NULL_PTR(out);

        try
        {
            Result<Tensor> cloned = tensor->implTensor.clone();
            MLT_CHECK_ERR(cloned);
            *out = new MLTtensor(MLT_MOVE_VALUE(cloned));

            return mltMakeErrorNone();
        }
        catch (...)
        {
            return mltMakeErrorInternal();
        }
    }

    MLTerror mltTensorShape(const MLTtensor *tensor, MLTshape *shape)
    {
        MLT_CHECK_NULL_PTR(tensor);
        MLT_CHECK_NULL_PTR(shape);

        try
        {
            std::vector<size_t> shapeVec = tensor->implTensor.shape();

            if (shapeVec.size() > MLT_MAX_DIMS)
                return mltErrorCreate(MLT_SHAPE_MISMATCH, "shape problem.");

            MLTshape resultShape{};
            resultShape.size = shapeVec.size();
            std::copy(shapeVec.begin(), shapeVec.end(), resultShape.dims);

            *shape = resultShape;

            return mltMakeErrorNone();
        }
        catch(...)
        {
            return mltMakeErrorInternal();
        }
    }

    MLTerror mltTensorGet(const MLTtensor *tensor, MLTshape shape, float *out)
    {
        MLT_CHECK_NULL_PTR(out);
        MLT_CHECK_NULL_PTR(tensor);

        try
        {
            Result<float> res = tensor->implTensor.at(std::span<const size_t>(shape.dims, shape.size));
            MLT_CHECK_ERR(res);

            *out = res.value();

            return mltMakeErrorNone();
        }
        catch (...)
        {
            return mltMakeErrorInternal();
        }
    }

    MLTerror mltTensorSet(MLTtensor *tensor, MLTshape shape, float val)
    {
        MLT_CHECK_NULL_PTR(tensor);

        try 
        {
            Result<float*> res = tensor->implTensor.at(std::span<const size_t>(shape.dims, shape.size));
            MLT_CHECK_ERR(res);
            
            float* value = res.value();
            *value = val;

            return mltMakeErrorNone();
        }
        catch (...)
        {
            return mltMakeErrorInternal();
        }
    }

    MLTerror mltTensorTranspose(const MLTtensor *tensor, MLTtensor **out)
    {
        MLT_CHECK_NULL_PTR(tensor);
        MLT_CHECK_NULL_PTR(out);

        try
        {
            Result<Tensor> result = tensor->implTensor.transpose();
            MLT_CHECK_ERR(result);
            *out = new MLTtensor(MLT_MOVE_VALUE(result));

            return mltMakeErrorNone();
        }
        catch (...)
        {
            return mltMakeErrorInternal();
        }
    }

    MLTerror mltTensorMatmul(const MLTtensor *a, const MLTtensor *b, MLTtensor **out)
    {
        MLT_CHECK_NULL_PTR(a);
        MLT_CHECK_NULL_PTR(b);
        MLT_CHECK_NULL_PTR(out);

        try 
        {
            Result<Tensor> res = a->implTensor.matmul(b->implTensor);
            MLT_CHECK_ERR(res);

            *out = new MLTtensor(res.value());
            return mltMakeErrorNone();
        }
        catch (...)
        {
            return mltMakeErrorInternal();
        }
    }

    MLTerror mltTensorMultiply(const MLTtensor *a, const MLTtensor *b, MLTtensor **out)
    {
        MLT_CHECK_NULL_PTR(a);
        MLT_CHECK_NULL_PTR(b);
        MLT_CHECK_NULL_PTR(out);

        try
        {
            Result<Tensor> result = a->implTensor.multiply(b->implTensor);
            MLT_CHECK_ERR(result);
            *out = new MLTtensor(MLT_MOVE_VALUE(result));
            
            return mltMakeErrorNone();
        }
        catch (...)
        {
            return mltMakeErrorInternal();
        }
    }

    MLTerror mltTensorAdd(const MLTtensor *a, const MLTtensor *b, MLTtensor **out)
    {
        MLT_CHECK_NULL_PTR(a);
        MLT_CHECK_NULL_PTR(b);
        MLT_CHECK_NULL_PTR(out);

        try
        {
            Result<Tensor> result = a->implTensor.add(b->implTensor);
            MLT_CHECK_ERR(result);
            *out = new MLTtensor(MLT_MOVE_VALUE(result));

            return mltMakeErrorNone();
        }
        catch (...)
        {
            return mltMakeErrorInternal();
        }
    }

    MLTerror mltTensorSub(const MLTtensor *a, const MLTtensor *b, MLTtensor **out)
    {
        MLT_CHECK_NULL_PTR(a);
        MLT_CHECK_NULL_PTR(b);
        MLT_CHECK_NULL_PTR(out);

        try
        {
            Result<Tensor> result = a->implTensor.sub(b->implTensor);
            MLT_CHECK_ERR(result);
            *out = new MLTtensor(MLT_MOVE_VALUE(result));

            return mltMakeErrorNone();
        }
        catch (...)
        {
            return mltMakeErrorInternal();
        }
    }
}
#endif
