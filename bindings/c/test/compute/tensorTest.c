#include "mlt/core/error.h"
#include <mlt/compute/tensor.h>
#include "../testdef.h"

void createShapeTensorTest()
{
    const char* testName = "createTensorTest";

    MLTshape shape = { .dims = {2, 2, 3, 4}, .size = 4};

    MLTtensor* tensor = NULL;
    MLTerror err = mltTensorFrom(shape, &tensor);
    
    ASSERT_STATUS_OK(testName, err.type, err.message);

cleanup:
    if (tensor)
        mltTensorDestroy(tensor);
}

void createBufTensorTest()
{
    const char* testName = "createBufTensorTest";

    MLTshape shape = { .dims = {2, 2}, .size = 2};
    const float buf[4] = { 1.f, 2.f, 3.f, 4.f};
    
    MLTtensor* tensor = NULL;
    MLTerror err = mltTensorFromBuff(buf, 4, shape, &tensor);

    ASSERT_STATUS_OK(testName, err.type, err.message);

    float one;
    float two;
    float three;
    float four;

    mltTensorGet(tensor, (MLTshape){ .dims = {0, 0}, .size = 2 }, &one);
    mltTensorGet(tensor, (MLTshape){ .dims = {0, 1}, .size = 2 }, &two);
    mltTensorGet(tensor, (MLTshape){ .dims = {1, 0}, .size = 2 }, &three);
    mltTensorGet(tensor, (MLTshape){ .dims = {1, 1}, .size = 2 }, &four);

    ASSERT_EQ_FLOAT(testName, one, buf[0], "Value doesnt match.");
    ASSERT_EQ_FLOAT(testName, two, buf[1], "Value doesnt match.");
    ASSERT_EQ_FLOAT(testName, three, buf[2], "Value doesnt match.");
    ASSERT_EQ_FLOAT(testName, four, buf[3], "Value doesnt match.");

cleanup:
    if (tensor)
        mltTensorDestroy(tensor);
}

int main(void)
{
    createShapeTensorTest();
    createBufTensorTest();
}
