#ifndef MLTOOLING_C_TENSOR_H
#define MLTOOLING_C_TENSOR_H

#include <stddef.h>
#include <mlt/core/error.h>

#ifdef __cplusplus
extern "C"
{
#endif

#define MLT_MAX_DIMS 6

typedef struct MLTshape
{
    size_t dims[MLT_MAX_DIMS];
    size_t size;
} MLTshape;

typedef struct MLTtensor MLTtensor;

MLTOOLING_C MLTerror mltTensorFrom(MLTshape, MLTtensor** out);
MLTOOLING_C MLTerror mltTensorFromBuff(const float* buf, size_t len, MLTshape shape, MLTtensor** out);
MLTOOLING_C void mltTensorDestroy(MLTtensor* tensor);

MLTOOLING_C MLTerror mltTensorCopy(const MLTtensor* tensor, MLTtensor** out);
// MLTOOLING_C MLTerror mltTensorClone(const MLTtensor* tensor, MLTtensor** out);
MLTOOLING_C MLTerror mltTensorShape(const MLTtensor* tensor, MLTshape* shape);
MLTOOLING_C MLTerror mltTensorGet(const MLTtensor* tensor, MLTshape shape, float* out);
MLTOOLING_C MLTerror mltTensorSet(MLTtensor* tensor, MLTshape shape, float val);
MLTOOLING_C MLTerror mltTensorTranspose(const MLTtensor* tensor, MLTtensor** out);

MLTOOLING_C MLTerror mltTensorMatmul(const MLTtensor* a, const MLTtensor* b, MLTtensor** out);
MLTOOLING_C MLTerror mltTensorMultiply(const MLTtensor* a, const MLTtensor* b, MLTtensor** out);
MLTOOLING_C MLTerror mltTensorAdd(const MLTtensor* a, const MLTtensor* b, MLTtensor** out);
MLTOOLING_C MLTerror mltTensorSub(const MLTtensor* a, const MLTtensor* b, MLTtensor** out);

#ifdef __cplusplus
}
#endif

#endif
