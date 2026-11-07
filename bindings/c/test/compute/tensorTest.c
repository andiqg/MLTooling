#include "mlt/core/error.h"
#include <mlt/compute/tensor.h>
#include "../testdef.h"

/* ------------------------------------------------------------------ helpers */

static MLTshape shape1(size_t a)
{
    return (MLTshape){ .dims = {a}, .size = 1 };
}

static MLTshape shape2(size_t a, size_t b)
{
    return (MLTshape){ .dims = {a, b}, .size = 2 };
}

/* Compares every element of t with expected (row-major order).
 * Returns 0 on success, 1 on failure. */
static int checkValues(const char* test, const MLTtensor* t,
                       const float* expected, size_t count, const char* msg)
{
    MLTshape shape;
    MLTerror err = mltTensorShape(t, &shape);
    if (assert_status_ok(test, err.type, err.message))
        return 1;

    size_t total = 1;
    for (size_t d = 0; d < shape.size; ++d)
        total *= shape.dims[d];

    if (assert_eq_int(test, (int)total, (int)count, "Element count doesnt match."))
        return 1;

    for (size_t flat = 0; flat < count; ++flat)
    {
        /* Convert the flat row-major index into a multi-index */
        MLTshape idx = shape;
        size_t rem = flat;
        for (size_t d = shape.size; d-- > 0;)
        {
            idx.dims[d] = rem % shape.dims[d];
            rem /= shape.dims[d];
        }

        float v = 0.f;
        err = mltTensorGet(t, idx, &v);
        if (assert_status_ok(test, err.type, err.message))
            return 1;
        if (assert_eq_float(test, v, expected[flat], msg))
            return 1;
    }
    return 0;
}

static int checkShape(const char* test, const MLTtensor* t, MLTshape expected)
{
    MLTshape s;
    MLTerror err = mltTensorShape(t, &s);
    if (assert_status_ok(test, err.type, err.message))
        return 1;
    if (assert_eq_int(test, (int)s.size, (int)expected.size, "Rank doesnt match."))
        return 1;
    for (size_t d = 0; d < expected.size; ++d)
        if (assert_eq_int(test, (int)s.dims[d], (int)expected.dims[d], "Dim doesnt match."))
            return 1;
    return 0;
}

typedef MLTerror (*BinaryFn)(const MLTtensor*, const MLTtensor*, MLTtensor**);

/* Builds a and b, applies fn and compares shape and values of the result. */
static int runBinary(const char* test, BinaryFn fn,
                     const float* aData, size_t aLen, MLTshape aShape,
                     const float* bData, size_t bLen, MLTshape bShape,
                     const float* expData, size_t expLen, MLTshape expShape)
{
    MLTtensor* a = NULL;
    MLTtensor* b = NULL;
    MLTtensor* r = NULL;
    int rc = 1;
    MLTerror err;

    err = mltTensorFromBuff(aData, aLen, aShape, &a);
    ASSERT_STATUS_OK(test, err.type, err.message);
    err = mltTensorFromBuff(bData, bLen, bShape, &b);
    ASSERT_STATUS_OK(test, err.type, err.message);
    err = fn(a, b, &r);
    ASSERT_STATUS_OK(test, err.type, err.message);

    if (checkShape(test, r, expShape)) goto cleanup;
    if (checkValues(test, r, expData, expLen, "Value doesnt match.")) goto cleanup;

    rc = 0;
cleanup:
    mltTensorDestroy(a);
    mltTensorDestroy(b);
    mltTensorDestroy(r);
    return rc;
}

/* Expects fn to report an error (type != 0). */
static int runBinaryExpectError(const char* test, BinaryFn fn,
                                const float* aData, size_t aLen, MLTshape aShape,
                                const float* bData, size_t bLen, MLTshape bShape)
{
    MLTtensor* a = NULL;
    MLTtensor* b = NULL;
    MLTtensor* r = NULL;
    int rc = 1;
    MLTerror err;

    err = mltTensorFromBuff(aData, aLen, aShape, &a);
    ASSERT_STATUS_OK(test, err.type, err.message);
    err = mltTensorFromBuff(bData, bLen, bShape, &b);
    ASSERT_STATUS_OK(test, err.type, err.message);

    err = fn(a, b, &r);
    ASSERT_EQ_INT(test, err.type != 0, 1, "Error not detected.");

    rc = 0;
cleanup:
    mltTensorDestroy(a);
    mltTensorDestroy(b);
    mltTensorDestroy(r);
    return rc;
}

/* Shared test data */
static const float A23[6] = {1, 2, 3, 4, 5, 6};
static const float B23[6] = {10, 20, 30, 40, 50, 60};
static const float V3[3]  = {10, 20, 30};
static const float S1[1]  = {2};

/* ---------------------------------------------------------------------- add */

void addSameShapeTest(void)
{
    const float exp[] = {11, 22, 33, 44, 55, 66};
    runBinary("addSameShapeTest", mltTensorAdd,
              A23, 6, shape2(2, 3), B23, 6, shape2(2, 3),
              exp, 6, shape2(2, 3));
}

void addScalarTest(void)
{
    const float exp[] = {3, 4, 5, 6, 7, 8};
    runBinary("addScalarTest", mltTensorAdd,
              A23, 6, shape2(2, 3), S1, 1, shape1(1),
              exp, 6, shape2(2, 3));
}

/* Only valid if the C API broadcasts trailing dims (dim 0) */
void addMatrixVectorTest(void)
{
    const float exp[] = {11, 22, 33, 14, 25, 36};
    runBinary("addMatrixVectorTest", mltTensorAdd,
              A23, 6, shape2(2, 3), V3, 3, shape1(3),
              exp, 6, shape2(2, 3));
}

void addShapeMismatchTest(void)
{
    const float other[] = {1, 2, 3, 4};
    runBinaryExpectError("addShapeMismatchTest", mltTensorAdd,
                         A23, 6, shape2(2, 3), other, 4, shape2(2, 2));
}

/* ---------------------------------------------------------------------- sub */

void subSameShapeTest(void)
{
    const float exp[] = {-9, -18, -27, -36, -45, -54};
    runBinary("subSameShapeTest", mltTensorSub,
              A23, 6, shape2(2, 3), B23, 6, shape2(2, 3),
              exp, 6, shape2(2, 3));
}

/* a - b != b - a */
void subReversedOrderTest(void)
{
    const float exp[] = {9, 18, 27, 36, 45, 54};
    runBinary("subReversedOrderTest", mltTensorSub,
              B23, 6, shape2(2, 3), A23, 6, shape2(2, 3),
              exp, 6, shape2(2, 3));
}

void subMatrixScalarTest(void)
{
    const float exp[] = {-1, 0, 1, 2, 3, 4};
    runBinary("subMatrixScalarTest", mltTensorSub,
              A23, 6, shape2(2, 3), S1, 1, shape1(1),
              exp, 6, shape2(2, 3));
}

/* scalar - matrix: operand order must be preserved */
void subScalarMatrixTest(void)
{
    const float exp[] = {1, 0, -1, -2, -3, -4};
    runBinary("subScalarMatrixTest", mltTensorSub,
              S1, 1, shape1(1), A23, 6, shape2(2, 3),
              exp, 6, shape2(2, 3));
}

/* vector - matrix, vector on the left */
void subVectorMatrixTest(void)
{
    const float exp[] = {9, 18, 27, 6, 15, 24};
    runBinary("subVectorMatrixTest", mltTensorSub,
              V3, 3, shape1(3), A23, 6, shape2(2, 3),
              exp, 6, shape2(2, 3));
}

void subShapeMismatchTest(void)
{
    const float other[] = {1, 2, 3, 4};
    runBinaryExpectError("subShapeMismatchTest", mltTensorSub,
                         A23, 6, shape2(2, 3), other, 4, shape2(2, 2));
}

/* ----------------------------------------------------------------- multiply */

void multiplySameShapeTest(void)
{
    const float exp[] = {10, 40, 90, 160, 250, 360};
    runBinary("multiplySameShapeTest", mltTensorMultiply,
              A23, 6, shape2(2, 3), B23, 6, shape2(2, 3),
              exp, 6, shape2(2, 3));
}

void multiplyScalarTest(void)
{
    const float exp[] = {2, 4, 6, 8, 10, 12};
    runBinary("multiplyScalarTest", mltTensorMultiply,
              A23, 6, shape2(2, 3), S1, 1, shape1(1),
              exp, 6, shape2(2, 3));
}

void multiplyMatrixVectorTest(void)
{
    const float exp[] = {10, 40, 90, 40, 100, 180};
    runBinary("multiplyMatrixVectorTest", mltTensorMultiply,
              A23, 6, shape2(2, 3), V3, 3, shape1(3),
              exp, 6, shape2(2, 3));
}

/* Hadamard product is commutative */
void multiplyVectorMatrixTest(void)
{
    const float exp[] = {10, 40, 90, 40, 100, 180};
    runBinary("multiplyVectorMatrixTest", mltTensorMultiply,
              V3, 3, shape1(3), A23, 6, shape2(2, 3),
              exp, 6, shape2(2, 3));
}

void multiplyShapeMismatchTest(void)
{
    const float other[] = {1, 2, 3, 4};
    runBinaryExpectError("multiplyShapeMismatchTest", mltTensorMultiply,
                         A23, 6, shape2(2, 3), other, 4, shape2(2, 2));
}

/* ------------------------------------------------------------------- matmul */

void matmulTest(void)
{
    const float b[]   = {7, 8, 9, 10, 11, 12};   /* (3,2) */
    const float exp[] = {58, 64, 139, 154};
    runBinary("matmulTest", mltTensorMatmul,
              A23, 6, shape2(2, 3), b, 6, shape2(3, 2),
              exp, 4, shape2(2, 2));
}

/* inner dimensions 3 and 2 do not match */
void matmulShapeMismatchTest(void)
{
    const float b[] = {1, 2, 3, 4};
    runBinaryExpectError("matmulShapeMismatchTest", mltTensorMatmul,
                         A23, 6, shape2(2, 3), b, 4, shape2(2, 2));
}

/* (2,3) * transpose((2,3)) -> (2,2) */
void matmulTransposedTest(void)
{
    const char* testName = "matmulTransposedTest";
    MLTtensor* a = NULL;
    MLTtensor* aT = NULL;
    MLTtensor* r = NULL;
    const float exp[] = {14, 32, 32, 77};
    MLTerror err;

    err = mltTensorFromBuff(A23, 6, shape2(2, 3), &a);
    ASSERT_STATUS_OK(testName, err.type, err.message);
    err = mltTensorTranspose(a, &aT);
    ASSERT_STATUS_OK(testName, err.type, err.message);
    err = mltTensorMatmul(a, aT, &r);
    ASSERT_STATUS_OK(testName, err.type, err.message);

    if (checkShape(testName, r, shape2(2, 2))) goto cleanup;
    checkValues(testName, r, exp, 4, "Value doesnt match.");

cleanup:
    mltTensorDestroy(a);
    mltTensorDestroy(aT);
    mltTensorDestroy(r);
}

/* ---------------------------------------------------------- tensor lifecycle */

void tensorGetSetTest(void)
{
    const char* testName = "tensorGetSetTest";
    MLTtensor* t = NULL;
    float v = 0.f;
    MLTerror err;

    err = mltTensorFromBuff(A23, 6, shape2(2, 3), &t);
    ASSERT_STATUS_OK(testName, err.type, err.message);

    err = mltTensorSet(t, shape2(1, 2), 99.f);
    ASSERT_STATUS_OK(testName, err.type, err.message);
    err = mltTensorGet(t, shape2(1, 2), &v);
    ASSERT_STATUS_OK(testName, err.type, err.message);
    ASSERT_EQ_FLOAT(testName, v, 99.f, "Value doesnt match after set.");

    /* neighbours must stay untouched */
    err = mltTensorGet(t, shape2(1, 1), &v);
    ASSERT_STATUS_OK(testName, err.type, err.message);
    ASSERT_EQ_FLOAT(testName, v, 5.f, "Neighbour was modified.");

cleanup:
    mltTensorDestroy(t);
}

void tensorGetOutOfBoundsTest(void)
{
    const char* testName = "tensorGetOutOfBoundsTest";
    MLTtensor* t = NULL;
    float v = 0.f;
    MLTerror err;

    err = mltTensorFromBuff(A23, 6, shape2(2, 3), &t);
    ASSERT_STATUS_OK(testName, err.type, err.message);

    err = mltTensorGet(t, shape2(2, 0), &v);
    ASSERT_EQ_INT(testName, err.type != 0, 1, "Row out of bounds not detected.");
    err = mltTensorGet(t, shape2(0, 3), &v);
    ASSERT_EQ_INT(testName, err.type != 0, 1, "Col out of bounds not detected.");

cleanup:
    mltTensorDestroy(t);
}

void tensorFromBuffWrongLengthTest(void)
{
    const char* testName = "tensorFromBuffWrongLengthTest";
    MLTtensor* t = NULL;

    /* buffer has 6 elements, shape (2,2) needs 4 */
    MLTerror err = mltTensorFromBuff(A23, 6, shape2(2, 2), &t);
    ASSERT_EQ_INT(testName, err.type != 0, 1, "Length mismatch not detected.");

cleanup:
    mltTensorDestroy(t);
}

void tensorCopyIsIndependentTest(void)
{
    const char* testName = "tensorCopyIsIndependentTest";
    MLTtensor* t = NULL;
    MLTtensor* c = NULL;
    float v = 0.f;
    MLTerror err;

    err = mltTensorFromBuff(A23, 6, shape2(2, 3), &t);
    ASSERT_STATUS_OK(testName, err.type, err.message);
    err = mltTensorCopy(t, &c);
    ASSERT_STATUS_OK(testName, err.type, err.message);

    if (checkValues(testName, c, A23, 6, "Copy has wrong value.")) goto cleanup;

    /* modifying the copy must not change the original */
    err = mltTensorSet(c, shape2(0, 0), 100.f);
    ASSERT_STATUS_OK(testName, err.type, err.message);
    err = mltTensorGet(t, shape2(0, 0), &v);
    ASSERT_STATUS_OK(testName, err.type, err.message);
    ASSERT_EQ_FLOAT(testName, v, 1.f, "Original was modified through copy.");

cleanup:
    mltTensorDestroy(t);
    mltTensorDestroy(c);
}

void tensorTransposeTest(void)
{
    const char* testName = "tensorTransposeTest";
    MLTtensor* t = NULL;
    MLTtensor* tr = NULL;
    const float exp[] = {1, 4, 2, 5, 3, 6};   /* (3,2) */
    MLTerror err;

    err = mltTensorFromBuff(A23, 6, shape2(2, 3), &t);
    ASSERT_STATUS_OK(testName, err.type, err.message);
    err = mltTensorTranspose(t, &tr);
    ASSERT_STATUS_OK(testName, err.type, err.message);

    if (checkShape(testName, tr, shape2(3, 2))) goto cleanup;
    checkValues(testName, tr, exp, 6, "Transposed value doesnt match.");

cleanup:
    mltTensorDestroy(t);
    mltTensorDestroy(tr);
}

int main(void)
{
    addSameShapeTest();
    addScalarTest();
    addMatrixVectorTest();
    addShapeMismatchTest();

    subSameShapeTest();
    subReversedOrderTest();
    subMatrixScalarTest();
    subScalarMatrixTest();
    subVectorMatrixTest();
    subShapeMismatchTest();

    multiplySameShapeTest();
    multiplyScalarTest();
    multiplyMatrixVectorTest();
    multiplyVectorMatrixTest();
    multiplyShapeMismatchTest();

    matmulTest();
    matmulShapeMismatchTest();
    matmulTransposedTest();

    tensorGetSetTest();
    tensorGetOutOfBoundsTest();
    tensorFromBuffWrongLengthTest();
    tensorCopyIsIndependentTest();
    tensorTransposeTest();

    return 0;
}
