#include <stdio.h>

#include <mlt/compute/tensor.h>

#define N 2

int main(void)
{
    MLTtensor* tensor;
    MLTshape shape = { {2, 3}, N };
    MLTerror err = mltTensorFrom(shape, &tensor);

    if (!(err.type == MLT_ERROR_NONE))
        printf("An error occured!\n");
    else
        printf("Tensor created!\n");

    mltTensorDestroy(tensor);   

    return 0;
}
