#include "../testdef.hpp"

#include <expected>
#include <span>

#include <mlt/macros.hpp>

import mlt.core.error;
import mlt.internal.compute.core.sizearray;
import mlt.internal.compute.core.mltarray;

using namespace mlt::core;
using namespace mlt::compute::core;

#define ERROR(var) var.error().msg.data()

static void MltArrayFromTest()
{
    DefaultSizeArray cShape = DefaultSizeArray({2, 2, 3, 4});
    DefaultSizeArray cStrides = DefaultSizeArray({24, 12, 4, 1});

    auto cArr = MltArray::from(cShape);

    if (!cArr)
        throwCustomMessage({ERROR(cArr)});

    const MltArray arr = MLT_MOVE_VALUE(cArr);

    assertEq(cShape.size(), cStrides.size(), "Strides have the wrong size.");

    for (size_t i = 0; i < arr.shape.size(); ++i)
        assertEq(arr.shape[i], cShape[i], "Shape was not initialized properly.");

    for (size_t i = 0; i < arr.strides.size(); ++i)
        assertEq(arr.strides[i], cStrides[i], "Strides were not initialized properly.");
}

static void MltArrayOpTest()
{
    const float testValues[] = { 1.f, 2.f, 3.f, 4.f };
    DefaultSizeArray cShape = DefaultSizeArray({ 2, 2});
    
    auto cArr = MltArray::from(std::span<const float>(testValues) , cShape);

    if (!cArr)
        throwCustomMessage({ERROR(cArr)});

    MltArray arr = MLT_MOVE_VALUE(cArr);
    
    assertEq(arr[0,0], 1.f, "Value doesnt match.");
    assertEq(arr[0,1], 2.f, "Value doesnt match.");
    assertEq(arr[1,0], 3.f, "Value doesnt match.");
    assertEq(arr[1,1], 4.f, "Value doesnt match.");
}

static void MltArrayAtTest()
{
    const float testValues[] = { 1.f, 2.f, 3.f, 4.f };
    
    DefaultSizeArray cShape = DefaultSizeArray({ 2, 2 });

    auto cArr = MltArray::from(std::span<const float>(testValues), cShape);

    if (!cArr)
        throwCustomMessage({ERROR(cArr)});

    MltArray arr = MLT_MOVE_VALUE(cArr);

    
    assertEq(**arr.at(0,0), 1.f, "Value doesnt match.");
    assertEq(**arr.at(0,1), 2.f, "Value doesnt match.");
    assertEq(**arr.at(1,0), 3.f, "Value doesnt match.");
    assertEq(**arr.at(1,1), 4.f, "Value doesnt match.");
}

static void MltArrayRowMajorTransposeTest()
{
    const float testValues[] = { 1.f, 2.f, 3.f, 4.f };
    DefaultSizeArray shape = DefaultSizeArray({2, 2});
    MltArray arr = MLT_MOVE_VALUE(MltArray::from(DefaultSizeArray({2, 2})));

    for (size_t i = 0; i < shape[0]; i++)
    {
        for (size_t j = 0; j < shape[1]; j++)
        {
            size_t linIdx = i * shape[1] + j;
            if (std::expected<float*, MltError> val = arr.at(i, j))
                **val = testValues[linIdx];
            else
                throwCustomMessage("at() failed while filling test array.");
        }
    }

    MltArray transposed = arr.transpose();

    for (size_t i = 0; i < shape[0]; i++)
    {
        for (size_t j = 0; j < shape[1]; j++)
        {
            // transposed(j, i) == arr(i, j)
            std::expected<float*, MltError> original = arr.at(i, j);
            std::expected<float*, MltError> transposedVal = transposed.at(j, i);

            if (!original)
                throwCustomMessage("at() failed reading original array.");
            if (!transposedVal)
                throwCustomMessage("at() failed reading transposed array.");

            assertEq(
                *transposedVal.value(),
                *original.value(),
                std::format("Transpose failed at ({},{}).", j, i)
            );
        }
    }
}

static void MltArrayRowMajor3DTransposeTest()
{
    DefaultSizeArray cShape = DefaultSizeArray({2, 3, 4});

    auto cArr = MltArray::from(cShape);

    if (!cArr)
        throwCustomMessage({ERROR(cArr)});

    MltArray arr = MLT_MOVE_VALUE(cArr);
    
    const size_t d0 = arr.shape[0], d1 = arr.shape[1], d2 = arr.shape[2];
    std::vector<float> testValues(d0 * d1 * d2);
    
    for (size_t idx = 0; idx < testValues.size(); ++idx)
        testValues[idx] = static_cast<float>(idx);

    for (size_t i = 0; i < d0; i++)
        for (size_t j = 0; j < d1; j++)
            for (size_t k = 0; k < d2; k++)
            {
                size_t linIdx = i * (d1 * d2) + j * d2 + k;
                if (std::expected<float*, MltError> val = arr.at(i, j, k))
                    *val.value() = testValues[linIdx];
                else
                    throwCustomMessage("at() failed while filling 3D test array.");
            }
    
    MltArray transposed = arr.transpose();

    for (size_t i = 0; i < d0; i++)
        for (size_t j = 0; j < d1; j++)
            for (size_t k = 0; k < d2; k++)
            {
                std::expected<float*, MltError> original = arr.at(i, j, k);
                std::expected<float*, MltError> transposedVal = transposed.at(k, j, i);

                if (!original)
                    throwCustomMessage("at() failed reading original 3D array.");
                if (!transposedVal)
                    throwCustomMessage("at() failed reading transposed 3D array.");

                assertEq(
                    *transposedVal.value(),
                    *original.value(),
                    std::format("3D transpose failed at ({},{},{}).", k, j, i)
                );
            }
}

static void MltArrayIsContigousTest()
{
    DefaultSizeArray cShape = DefaultSizeArray({3, 4});

    auto cArr = MltArray::from(cShape);

    if (!cArr)
        throwCustomMessage({ERROR(cArr)});

    MltArray arr = MLT_MOVE_VALUE(cArr);
    MltArray transposed = arr.transpose();

    assertEq(false, transposed.isContiguous(), "Contigous doesnt work.");
}

REGISTER_TEST(MltArrayFromTest)
REGISTER_TEST(MltArrayOpTest);
REGISTER_TEST(MltArrayAtTest)
REGISTER_TEST(MltArrayRowMajorTransposeTest)
REGISTER_TEST(MltArrayRowMajor3DTransposeTest)
REGISTER_TEST(MltArrayIsContigousTest)
