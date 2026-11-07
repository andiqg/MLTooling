#include "../../testdef.hpp"

#include <mlt/macros.hpp>

#include <span>

import mlt.core.error;
import mlt.internal.compute.operations;
import mlt.internal.compute.core.mltarray;
import mlt.internal.compute.core.sizearray;

using namespace mlt::core;
using namespace mlt::compute::ops;
using namespace mlt::compute::core;

#define ERROR(var) var.error().msg.data()

constexpr float aData[] = {
    1.f, 2.f, 3.f,
    4.f, 5.f, 6.f
};

constexpr float bData[] = {
    7.f, 8.f,
    9.f, 10.f,
    11.f, 12.f
};

void assertValues(const MltArray& result, std::span<const float> expected, const char* name)
{
    const float* C = reinterpret_cast<const float*>(result.data->data);
    for (size_t i = 0; i < expected.size(); ++i)
        assertEq(C[i], expected[i], name);
}

void testOperationsMatmul()
{
    DefaultSizeArray aShape = DefaultSizeArray({2, 3});
    DefaultSizeArray bShape = DefaultSizeArray({3, 2});
    MltArray a = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(aData), aShape));
    MltArray b = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(bData), bShape));

    MltArray result = MLT_MOVE_VALUE(matmul(a, b));

    assertEq(result.shape.size(), (size_t)2, "Wrong shape size");
    assertEq(result.shape[0], (size_t)2, "Wrong row shape");
    assertEq(result.shape[1], (size_t)2, "Wrong col shape");

    const float* C = reinterpret_cast<const float*>(result.data->data);

    assertEq(C[0], 58.f, "Wrong value[0,0]");
    assertEq(C[1], 64.f, "Wrong value [0,1]");
    assertEq(C[2], 139.f, "Wrong value [1,0]");
    assertEq(C[3], 154.f, "Wrong value [1,1]");
}

void testOperationsMatmulTransposed()
{
    DefaultSizeArray aShape = DefaultSizeArray({2, 3});
    DefaultSizeArray bShape = DefaultSizeArray({2, 3});

    Result<MltArray> cA = MltArray::from(std::span<const float>(aData), aShape);
    Result<MltArray> cB = MltArray::from(std::span<const float>(bData), bShape);

    if (!cA)
        throwCustomMessage({ERROR(cA)});
    
    if (!cB)
        throwCustomMessage({ERROR(cB)});

    MltArray a = MLT_MOVE_VALUE(cA);
    MltArray b = MLT_MOVE_VALUE(cB);

    Result<MltArray> cBTrans = b.transpose();

    if (!cBTrans)
        throwCustomMessage({ERROR(cBTrans)});
    
    MltArray bTrans = MLT_MOVE_VALUE(cBTrans);

    Result<MltArray> cResult = matmul(a, bTrans);

    if (!cResult)
        throwCustomMessage({ERROR(cResult)});
    
    MltArray result = MLT_MOVE_VALUE(cResult);

    assertEq(result.shape.size(), (size_t)2, "Wrong shape size");
    assertEq(result.shape[0], (size_t)2, "Wrong row shape");
    assertEq(result.shape[1], (size_t)2, "Wrong col shape");

    const float* C = reinterpret_cast<const float*>(result.data->data);

    assertEq(C[0], 58.f, "Wrong value[0,0]");
    assertEq(C[1], 64.f, "Wrong value [0,1]");
    assertEq(C[2], 139.f, "Wrong value [1,0]");
    assertEq(C[3], 154.f, "Wrong value [1,1]");
}



void testOperationsMultiplySameShape()
{
    const float aData[] = {1, 2, 3, 4, 5, 6};
    const float bData[] = {10, 20, 30, 40, 50, 60};
    DefaultSizeArray shape = DefaultSizeArray({2, 3});

    MltArray a = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(aData), shape));
    MltArray b = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(bData), shape));

    MltArray result = MLT_MOVE_VALUE(multiply(a, b, 0));

    assertEq(result.shape.size(), (size_t)2, "Wrong shape size");
    assertEq(result.shape[0], (size_t)2, "Wrong row shape");
    assertEq(result.shape[1], (size_t)3, "Wrong col shape");

    const float expected[] = {10, 40, 90, 160, 250, 360};
    assertValues(result, expected, "Wrong same-shape value");
}

void testOperationsMultiplyMatrixVectorDim0()
{
    // A (2,3) * v (3), dim 0 -> result[i,j] = A[i,j] * v[j]
    const float aData[] = {1, 2, 3, 4, 5, 6};
    const float vData[] = {10, 20, 30};

    MltArray a = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(aData), DefaultSizeArray({2, 3})));
    MltArray v = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(vData), DefaultSizeArray({3})));

    MltArray result = MLT_MOVE_VALUE(multiply(a, v, 0));

    assertEq(result.shape.size(), (size_t)2, "Wrong shape size");
    assertEq(result.shape[0], (size_t)2, "Wrong row shape");
    assertEq(result.shape[1], (size_t)3, "Wrong col shape");

    const float expected[] = {10, 40, 90, 40, 100, 180};
    assertValues(result, expected, "Wrong matrix*vector dim0 value");
}

void testOperationsMultiplyMatrixVectorDim1()
{
    // A (2,3) * v (2), dim 1 -> result[i,j] = A[i,j] * v[i]
    const float aData[] = {1, 2, 3, 4, 5, 6};
    const float vData[] = {10, 100};

    MltArray a = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(aData), DefaultSizeArray({2, 3})));
    MltArray v = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(vData), DefaultSizeArray({2})));

    MltArray result = MLT_MOVE_VALUE(multiply(a, v, 1));

    assertEq(result.shape[0], (size_t)2, "Wrong row shape");
    assertEq(result.shape[1], (size_t)3, "Wrong col shape");

    const float expected[] = {10, 20, 30, 400, 500, 600};
    assertValues(result, expected, "Wrong matrix*vector dim1 value");
}

void testOperationsMultiplyVectorMatrixCommutative()
{
    // multiply(v, A, 0) must be the same as multiply(A, v, 0)
    const float aData[] = {1, 2, 3, 4, 5, 6};
    const float vData[] = {10, 20, 30};

    MltArray a = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(aData), DefaultSizeArray({2, 3})));
    MltArray v = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(vData), DefaultSizeArray({3})));

    MltArray result = MLT_MOVE_VALUE(multiply(v, a, 0));

    assertEq(result.shape.size(), (size_t)2, "Wrong shape size");
    assertEq(result.shape[0], (size_t)2, "Wrong row shape");
    assertEq(result.shape[1], (size_t)3, "Wrong col shape");

    const float expected[] = {10, 40, 90, 40, 100, 180};
    assertValues(result, expected, "Wrong vector*matrix value");
}

void testOperationsMultiplyVectorVector()
{
    const float uData[] = {1, 2, 3};
    const float vData[] = {4, 5, 6};

    MltArray u = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(uData), DefaultSizeArray({3})));
    MltArray v = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(vData), DefaultSizeArray({3})));

    MltArray result = MLT_MOVE_VALUE(multiply(u, v, 0));

    assertEq(result.shape.size(), (size_t)1, "Wrong shape size");
    assertEq(result.shape[0], (size_t)3, "Wrong shape");

    const float expected[] = {4, 10, 18};
    assertValues(result, expected, "Wrong vector*vector value");
}

void testOperationsMultiplyTensorMatrix()
{
    // A (2,2,3) * B (2,3), dim 0 -> result[a,b,c] = A[a,b,c] * B[b,c]
    const float aData[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const float bData[] = {1, 2, 3, 4, 5, 6};

    MltArray a = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(aData), DefaultSizeArray({2, 2, 3})));
    MltArray b = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(bData), DefaultSizeArray({2, 3})));

    MltArray result = MLT_MOVE_VALUE(multiply(a, b, 0));

    assertEq(result.shape.size(), (size_t)3, "Wrong shape size");
    assertEq(result.shape[0], (size_t)2, "Wrong batch shape");
    assertEq(result.shape[1], (size_t)2, "Wrong row shape");
    assertEq(result.shape[2], (size_t)3, "Wrong col shape");

    const float expected[] = {1, 4, 9, 16, 25, 36, 7, 16, 27, 40, 55, 72};
    assertValues(result, expected, "Wrong tensor*matrix value");
}

void testOperationsMultiplyTensorVectorDim1()
{
    // A (2,2,3) * v (2), dim 1 -> result[a,b,c] = A[a,b,c] * v[b]
    // (outer = 2, mid = 2, inner = 3
    const float aData[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const float vData[] = {10, 100};

    MltArray a = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(aData), DefaultSizeArray({2, 2, 3})));
    MltArray v = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(vData), DefaultSizeArray({2})));

    MltArray result = MLT_MOVE_VALUE(multiply(a, v, 1));

    assertEq(result.shape.size(), (size_t)3, "Wrong shape size");

    const float expected[] = {10, 20, 30, 400, 500, 600, 70, 80, 90, 1000, 1100, 1200};
    assertValues(result, expected, "Wrong tensor*vector dim1 value");
}

void testOperationsMultiplyScalar()
{
    const float aData[] = {1, 2, 3, 4, 5, 6};
    const float sData[] = {2};

    MltArray a = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(aData), DefaultSizeArray({2, 3})));
    MltArray s = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(sData), DefaultSizeArray({1})));

    MltArray result = MLT_MOVE_VALUE(multiply(a, s, 0));

    assertEq(result.shape[0], (size_t)2, "Wrong row shape");
    assertEq(result.shape[1], (size_t)3, "Wrong col shape");

    const float expected[] = {2, 4, 6, 8, 10, 12};
    assertValues(result, expected, "Wrong scalar value");
}

void testOperationsMultiplyErrors()
{
    const float aData[] = {1, 2, 3, 4, 5, 6};
    const float vData[] = {1, 2};

    MltArray a = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(aData), DefaultSizeArray({2, 3})));
    MltArray v = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(vData), DefaultSizeArray({2})));

    assertEq(multiply(a, v, 0).has_value(), false, "Shape mismatch not detected");

    assertEq(multiply(a, v, 2).has_value(), false, "Invalid dim not detected");

    // dtype mismatch
    const double dData[] = {1, 2, 3, 4, 5, 6};
    MltArray d = MLT_MOVE_VALUE(MltArray::from(std::span<const double>(dData), DefaultSizeArray({2, 3})));
    assertEq(multiply(a, d, 0).has_value(), false, "Type mismatch not detected");
}

// ---------------------------------------------------------------- ADD

void testOperationsAddSameShape()
{
    const float aData[] = {1, 2, 3, 4, 5, 6};
    const float bData[] = {10, 20, 30, 40, 50, 60};
    DefaultSizeArray shape = DefaultSizeArray({2, 3});

    MltArray a = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(aData), shape));
    MltArray b = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(bData), shape));

    MltArray result = MLT_MOVE_VALUE(add(a, b, 0));

    assertEq(result.shape.size(), (size_t)2, "Wrong shape size");
    assertEq(result.shape[0], (size_t)2, "Wrong row shape");
    assertEq(result.shape[1], (size_t)3, "Wrong col shape");

    const float expected[] = {11, 22, 33, 44, 55, 66};
    assertValues(result, expected, "Wrong add same-shape value");
}

void testOperationsAddMatrixVectorDim0()
{
    // A (2,3) + v (3), dim 0 -> result[i,j] = A[i,j] + v[j]
    const float aData[] = {1, 2, 3, 4, 5, 6};
    const float vData[] = {10, 20, 30};

    MltArray a = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(aData), DefaultSizeArray({2, 3})));
    MltArray v = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(vData), DefaultSizeArray({3})));

    MltArray result = MLT_MOVE_VALUE(add(a, v, 0));

    assertEq(result.shape.size(), (size_t)2, "Wrong shape size");
    assertEq(result.shape[0], (size_t)2, "Wrong row shape");
    assertEq(result.shape[1], (size_t)3, "Wrong col shape");

    const float expected[] = {11, 22, 33, 14, 25, 36};
    assertValues(result, expected, "Wrong add matrix+vector dim0 value");
}

void testOperationsAddMatrixVectorDim1()
{
    // A (2,3) + v (2), dim 1 -> result[i,j] = A[i,j] + v[i]
    const float aData[] = {1, 2, 3, 4, 5, 6};
    const float vData[] = {10, 100};

    MltArray a = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(aData), DefaultSizeArray({2, 3})));
    MltArray v = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(vData), DefaultSizeArray({2})));

    MltArray result = MLT_MOVE_VALUE(add(a, v, 1));

    assertEq(result.shape[0], (size_t)2, "Wrong row shape");
    assertEq(result.shape[1], (size_t)3, "Wrong col shape");

    const float expected[] = {11, 12, 13, 104, 105, 106};
    assertValues(result, expected, "Wrong add matrix+vector dim1 value");
}

void testOperationsAddVectorMatrixCommutative()
{
    // add(v, A, 0) must be the same as add(A, v, 0)
    const float aData[] = {1, 2, 3, 4, 5, 6};
    const float vData[] = {10, 20, 30};

    MltArray a = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(aData), DefaultSizeArray({2, 3})));
    MltArray v = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(vData), DefaultSizeArray({3})));

    MltArray result = MLT_MOVE_VALUE(add(v, a, 0));

    assertEq(result.shape.size(), (size_t)2, "Wrong shape size");
    assertEq(result.shape[0], (size_t)2, "Wrong row shape");
    assertEq(result.shape[1], (size_t)3, "Wrong col shape");

    const float expected[] = {11, 22, 33, 14, 25, 36};
    assertValues(result, expected, "Wrong add vector+matrix value");
}

void testOperationsAddVectorVector()
{
    const float uData[] = {1, 2, 3};
    const float vData[] = {10, 20, 30};

    MltArray u = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(uData), DefaultSizeArray({3})));
    MltArray v = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(vData), DefaultSizeArray({3})));

    MltArray result = MLT_MOVE_VALUE(add(u, v, 0));

    assertEq(result.shape.size(), (size_t)1, "Wrong shape size");
    assertEq(result.shape[0], (size_t)3, "Wrong shape");

    const float expected[] = {11, 22, 33};
    assertValues(result, expected, "Wrong add vector+vector value");
}

void testOperationsAddTensorMatrix()
{
    // A (2,2,3) + B (2,3), dim 0 -> result[a,b,c] = A[a,b,c] + B[b,c]
    const float aData[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const float bData[] = {1, 2, 3, 4, 5, 6};

    MltArray a = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(aData), DefaultSizeArray({2, 2, 3})));
    MltArray b = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(bData), DefaultSizeArray({2, 3})));

    MltArray result = MLT_MOVE_VALUE(add(a, b, 0));

    assertEq(result.shape.size(), (size_t)3, "Wrong shape size");
    assertEq(result.shape[0], (size_t)2, "Wrong batch shape");
    assertEq(result.shape[1], (size_t)2, "Wrong row shape");
    assertEq(result.shape[2], (size_t)3, "Wrong col shape");

    const float expected[] = {2, 4, 6, 8, 10, 12, 8, 10, 12, 14, 16, 18};
    assertValues(result, expected, "Wrong add tensor+matrix value");
}

void testOperationsAddTensorVectorDim1()
{
    // A (2,2,3) + v (2), dim 1 -> result[a,b,c] = A[a,b,c] + v[b]
    // (outer = 2, mid = 2, inner = 3)
    const float aData[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const float vData[] = {10, 100};

    MltArray a = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(aData), DefaultSizeArray({2, 2, 3})));
    MltArray v = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(vData), DefaultSizeArray({2})));

    MltArray result = MLT_MOVE_VALUE(add(a, v, 1));

    assertEq(result.shape.size(), (size_t)3, "Wrong shape size");

    const float expected[] = {11, 12, 13, 104, 105, 106, 17, 18, 19, 110, 111, 112};
    assertValues(result, expected, "Wrong add tensor+vector dim1 value");
}

void testOperationsAddScalar()
{
    const float aData[] = {1, 2, 3, 4, 5, 6};
    const float sData[] = {2};

    MltArray a = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(aData), DefaultSizeArray({2, 3})));
    MltArray s = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(sData), DefaultSizeArray({1})));

    MltArray result = MLT_MOVE_VALUE(add(a, s, 0));

    assertEq(result.shape[0], (size_t)2, "Wrong row shape");
    assertEq(result.shape[1], (size_t)3, "Wrong col shape");

    const float expected[] = {3, 4, 5, 6, 7, 8};
    assertValues(result, expected, "Wrong add scalar value");
}

void testOperationsAddErrors()
{
    const float aData[] = {1, 2, 3, 4, 5, 6};
    const float vData[] = {1, 2};

    MltArray a = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(aData), DefaultSizeArray({2, 3})));
    MltArray v = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(vData), DefaultSizeArray({2})));

    assertEq(add(a, v, 0).has_value(), false, "Shape mismatch not detected");
    assertEq(add(a, v, 2).has_value(), false, "Invalid dim not detected");

    const double dData[] = {1, 2, 3, 4, 5, 6};
    MltArray d = MLT_MOVE_VALUE(MltArray::from(std::span<const double>(dData), DefaultSizeArray({2, 3})));
    assertEq(add(a, d, 0).has_value(), false, "Type mismatch not detected");
}

// ---------------------------------------------------------------- SUB

void testOperationsSubSameShape()
{
    const float aData[] = {1, 2, 3, 4, 5, 6};
    const float bData[] = {10, 20, 30, 40, 50, 60};
    DefaultSizeArray shape = DefaultSizeArray({2, 3});

    MltArray a = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(aData), shape));
    MltArray b = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(bData), shape));

    MltArray result = MLT_MOVE_VALUE(sub(a, b, 0));

    assertEq(result.shape.size(), (size_t)2, "Wrong shape size");
    assertEq(result.shape[0], (size_t)2, "Wrong row shape");
    assertEq(result.shape[1], (size_t)3, "Wrong col shape");

    const float expected[] = {-9, -18, -27, -36, -45, -54};
    assertValues(result, expected, "Wrong sub same-shape value");
}

void testOperationsSubNotCommutative()
{
    // a - b != b - identical shape
    const float aData[] = {1, 2, 3, 4, 5, 6};
    const float bData[] = {10, 20, 30, 40, 50, 60};
    DefaultSizeArray shape = DefaultSizeArray({2, 3});

    MltArray a = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(aData), shape));
    MltArray b = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(bData), shape));

    MltArray result = MLT_MOVE_VALUE(sub(b, a, 0));

    const float expected[] = {9, 18, 27, 36, 45, 54};
    assertValues(result, expected, "Wrong sub b-a value");
}

void testOperationsSubMatrixVectorDim0()
{
    // A (2,3) - v (3), dim 0 -> result[i,j] = A[i,j] - v[j]
    const float aData[] = {1, 2, 3, 4, 5, 6};
    const float vData[] = {10, 20, 30};

    MltArray a = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(aData), DefaultSizeArray({2, 3})));
    MltArray v = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(vData), DefaultSizeArray({3})));

    MltArray result = MLT_MOVE_VALUE(sub(a, v, 0));

    assertEq(result.shape.size(), (size_t)2, "Wrong shape size");
    assertEq(result.shape[0], (size_t)2, "Wrong row shape");
    assertEq(result.shape[1], (size_t)3, "Wrong col shape");

    const float expected[] = {-9, -18, -27, -6, -15, -24};
    assertValues(result, expected, "Wrong sub matrix-vector dim0 value");
}

void testOperationsSubMatrixVectorDim1()
{
    // A (2,3) - v (2), dim 1 -> result[i,j] = A[i,j] - v[i]
    const float aData[] = {1, 2, 3, 4, 5, 6};
    const float vData[] = {10, 100};

    MltArray a = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(aData), DefaultSizeArray({2, 3})));
    MltArray v = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(vData), DefaultSizeArray({2})));

    MltArray result = MLT_MOVE_VALUE(sub(a, v, 1));

    assertEq(result.shape[0], (size_t)2, "Wrong row shape");
    assertEq(result.shape[1], (size_t)3, "Wrong col shape");

    const float expected[] = {-9, -8, -7, -96, -95, -94};
    assertValues(result, expected, "Wrong sub matrix-vector dim1 value");
}

void testOperationsSubVectorMatrixOrder()
{
    // v (3) - A (2,3), dim 0 -> result[i,j] = v[j] - A[i,j]
    const float aData[] = {1, 2, 3, 4, 5, 6};
    const float vData[] = {10, 20, 30};

    MltArray a = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(aData), DefaultSizeArray({2, 3})));
    MltArray v = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(vData), DefaultSizeArray({3})));

    MltArray result = MLT_MOVE_VALUE(sub(v, a, 0));

    assertEq(result.shape.size(), (size_t)2, "Wrong shape size");
    assertEq(result.shape[0], (size_t)2, "Wrong row shape");
    assertEq(result.shape[1], (size_t)3, "Wrong col shape");

    const float expected[] = {9, 18, 27, 6, 15, 24};
    assertValues(result, expected, "Wrong sub vector-matrix value");
}

void testOperationsSubVectorVector()
{
    const float uData[] = {1, 2, 3};
    const float vData[] = {10, 20, 30};

    MltArray u = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(uData), DefaultSizeArray({3})));
    MltArray v = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(vData), DefaultSizeArray({3})));

    MltArray result = MLT_MOVE_VALUE(sub(u, v, 0));

    assertEq(result.shape.size(), (size_t)1, "Wrong shape size");
    assertEq(result.shape[0], (size_t)3, "Wrong shape");

    const float expected[] = {-9, -18, -27};
    assertValues(result, expected, "Wrong sub vector-vector value");
}

void testOperationsSubTensorMatrix()
{
    // A (2,2,3) - B (2,3), dim 0 -> result[a,b,c] = A[a,b,c] - B[b,c]
    const float aData[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const float bData[] = {1, 2, 3, 4, 5, 6};

    MltArray a = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(aData), DefaultSizeArray({2, 2, 3})));
    MltArray b = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(bData), DefaultSizeArray({2, 3})));

    MltArray result = MLT_MOVE_VALUE(sub(a, b, 0));

    assertEq(result.shape.size(), (size_t)3, "Wrong shape size");
    assertEq(result.shape[0], (size_t)2, "Wrong batch shape");
    assertEq(result.shape[1], (size_t)2, "Wrong row shape");
    assertEq(result.shape[2], (size_t)3, "Wrong col shape");

    const float expected[] = {0, 0, 0, 0, 0, 0, 6, 6, 6, 6, 6, 6};
    assertValues(result, expected, "Wrong sub tensor-matrix value");
}

void testOperationsSubMatrixTensorOrder()
{
    // B (2,3) - A (2,2,3), dim 0 -> result[a,b,c] = B[b,c] - A[a,b,c]
    const float aData[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const float bData[] = {1, 2, 3, 4, 5, 6};

    MltArray a = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(aData), DefaultSizeArray({2, 2, 3})));
    MltArray b = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(bData), DefaultSizeArray({2, 3})));

    MltArray result = MLT_MOVE_VALUE(sub(b, a, 0));

    assertEq(result.shape.size(), (size_t)3, "Wrong shape size");

    const float expected[] = {0, 0, 0, 0, 0, 0, -6, -6, -6, -6, -6, -6};
    assertValues(result, expected, "Wrong sub matrix-tensor value");
}

void testOperationsSubTensorVectorDim1()
{
    // A (2,2,3) - v (2), dim 1 -> result[a,b,c] = A[a,b,c] - v[b]
    const float aData[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const float vData[] = {10, 100};

    MltArray a = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(aData), DefaultSizeArray({2, 2, 3})));
    MltArray v = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(vData), DefaultSizeArray({2})));

    MltArray result = MLT_MOVE_VALUE(sub(a, v, 1));

    assertEq(result.shape.size(), (size_t)3, "Wrong shape size");

    const float expected[] = {-9, -8, -7, -96, -95, -94, -3, -2, -1, -90, -89, -88};
    assertValues(result, expected, "Wrong sub tensor-vector dim1 value");
}

void testOperationsSubVectorTensorDim1Order()
{
    // v (2) - A (2,2,3), dim 1 -> result[a,b,c] = v[b] - A[a,b,c]
    const float aData[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const float vData[] = {10, 100};

    MltArray a = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(aData), DefaultSizeArray({2, 2, 3})));
    MltArray v = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(vData), DefaultSizeArray({2})));

    MltArray result = MLT_MOVE_VALUE(sub(v, a, 1));

    assertEq(result.shape.size(), (size_t)3, "Wrong shape size");

    const float expected[] = {9, 8, 7, 96, 95, 94, 3, 2, 1, 90, 89, 88};
    assertValues(result, expected, "Wrong sub vector-tensor dim1 value");
}

void testOperationsSubMatrixScalar()
{
    // A - s
    const float aData[] = {1, 2, 3, 4, 5, 6};
    const float sData[] = {2};

    MltArray a = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(aData), DefaultSizeArray({2, 3})));
    MltArray s = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(sData), DefaultSizeArray({1})));

    MltArray result = MLT_MOVE_VALUE(sub(a, s, 0));

    assertEq(result.shape[0], (size_t)2, "Wrong row shape");
    assertEq(result.shape[1], (size_t)3, "Wrong col shape");

    const float expected[] = {-1, 0, 1, 2, 3, 4};
    assertValues(result, expected, "Wrong sub matrix-scalar value");
}

void testOperationsSubScalarMatrixOrder()
{
    const float aData[] = {1, 2, 3, 4, 5, 6};
    const float sData[] = {2};

    MltArray a = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(aData), DefaultSizeArray({2, 3})));
    MltArray s = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(sData), DefaultSizeArray({1})));

    MltArray result = MLT_MOVE_VALUE(sub(s, a, 0));

    assertEq(result.shape[0], (size_t)2, "Wrong row shape");
    assertEq(result.shape[1], (size_t)3, "Wrong col shape");

    const float expected[] = {1, 0, -1, -2, -3, -4};
    assertValues(result, expected, "Wrong sub scalar-matrix value");
}

void testOperationsSubScalarVectorOrder()
{
    const float vData[] = {1, 2, 3};
    const float sData[] = {10};

    MltArray v = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(vData), DefaultSizeArray({3})));
    MltArray s = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(sData), DefaultSizeArray({1})));

    MltArray result = MLT_MOVE_VALUE(sub(s, v, 0));

    assertEq(result.shape.size(), (size_t)1, "Wrong shape size");
    assertEq(result.shape[0], (size_t)3, "Wrong shape");

    const float expected[] = {9, 8, 7};
    assertValues(result, expected, "Wrong sub scalar-vector value");
}

void testOperationsSubErrors()
{
    const float aData[] = {1, 2, 3, 4, 5, 6};
    const float vData[] = {1, 2};

    MltArray a = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(aData), DefaultSizeArray({2, 3})));
    MltArray v = MLT_MOVE_VALUE(MltArray::from(std::span<const float>(vData), DefaultSizeArray({2})));

    assertEq(sub(a, v, 0).has_value(), false, "Shape mismatch not detected");
    assertEq(sub(a, v, 2).has_value(), false, "Invalid dim not detected");

    const double dData[] = {1, 2, 3, 4, 5, 6};
    MltArray d = MLT_MOVE_VALUE(MltArray::from(std::span<const double>(dData), DefaultSizeArray({2, 3})));
    assertEq(sub(a, d, 0).has_value(), false, "Type mismatch not detected");
}

REGISTER_TEST(testOperationsMatmul);
REGISTER_TEST(testOperationsMatmulTransposed);

REGISTER_TEST(testOperationsMultiplySameShape);
REGISTER_TEST(testOperationsMultiplyMatrixVectorDim0);
REGISTER_TEST(testOperationsMultiplyMatrixVectorDim1);
REGISTER_TEST(testOperationsMultiplyVectorMatrixCommutative);
REGISTER_TEST(testOperationsMultiplyVectorVector);
REGISTER_TEST(testOperationsMultiplyTensorMatrix);
REGISTER_TEST(testOperationsMultiplyTensorVectorDim1);
REGISTER_TEST(testOperationsMultiplyScalar);
REGISTER_TEST(testOperationsMultiplyErrors);

REGISTER_TEST(testOperationsAddSameShape);
REGISTER_TEST(testOperationsAddMatrixVectorDim0);
REGISTER_TEST(testOperationsAddMatrixVectorDim1);
REGISTER_TEST(testOperationsAddVectorMatrixCommutative);
REGISTER_TEST(testOperationsAddVectorVector);
REGISTER_TEST(testOperationsAddTensorMatrix);
REGISTER_TEST(testOperationsAddTensorVectorDim1);
REGISTER_TEST(testOperationsAddScalar);
REGISTER_TEST(testOperationsAddErrors);

REGISTER_TEST(testOperationsSubSameShape);
REGISTER_TEST(testOperationsSubNotCommutative);
REGISTER_TEST(testOperationsSubMatrixVectorDim0);
REGISTER_TEST(testOperationsSubMatrixVectorDim1);
REGISTER_TEST(testOperationsSubVectorMatrixOrder);
REGISTER_TEST(testOperationsSubVectorVector);
REGISTER_TEST(testOperationsSubTensorMatrix);
REGISTER_TEST(testOperationsSubMatrixTensorOrder);
REGISTER_TEST(testOperationsSubTensorVectorDim1);
REGISTER_TEST(testOperationsSubVectorTensorDim1Order);
REGISTER_TEST(testOperationsSubMatrixScalar);
REGISTER_TEST(testOperationsSubScalarMatrixOrder);
REGISTER_TEST(testOperationsSubScalarVectorOrder);
REGISTER_TEST(testOperationsSubErrors);
