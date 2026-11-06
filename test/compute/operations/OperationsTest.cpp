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

namespace
{
    // Vergleicht result.data mit expected, spart die Wiederholung der assertEq-Zeilen
    void assertValues(const MltArray& result, std::span<const float> expected, const char* name)
    {
        const float* C = reinterpret_cast<const float*>(result.data->data);
        for (size_t i = 0; i < expected.size(); ++i)
            assertEq(C[i], expected[i], name);
    }
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
    // A (2,3) * v (3), dim 0 -> result[i,j] = A[i,j] * v[j]  (Spalten skalieren)
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
    // A (2,3) * v (2), dim 1 -> result[i,j] = A[i,j] * v[i]  (Zeilen skalieren)
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
    // multiply(v, A, 0) muss identisch zu multiply(A, v, 0) sein
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
    // (outer = 2, mid = 2, inner = 3, testet alle drei Blöcke des Kernels)
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
    // Annahme: Skalar = Array mit einem Element. Passe die Shape an,
    // falls dein isScalar() etwas anderes erwartet (z.B. leere Shape).
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

    // v (2) passt bei dim 0 nicht auf die letzte Achse (3)
    assertEq(multiply(a, v, 0).has_value(), false, "Shape mismatch not detected");

    // dim zu groß: dim + op.rank() > host.rank()
    assertEq(multiply(a, v, 2).has_value(), false, "Invalid dim not detected");

    // dtype mismatch
    const double dData[] = {1, 2, 3, 4, 5, 6};
    MltArray d = MLT_MOVE_VALUE(MltArray::from(std::span<const double>(dData), DefaultSizeArray({2, 3})));
    assertEq(multiply(a, d, 0).has_value(), false, "Type mismatch not detected");
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
