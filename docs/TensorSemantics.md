# Broadcasting

MLTooling supports explicit broadcasting for selected element-wise operations.

Broadcasting is never performed implicitly by operators such as `operator*`.
When broadcasting is required, the corresponding operation must be called explicitly and a dimension must be specified.

```cpp
multiply(a, b, dim);
```

## Dimension

The `dim` parameter specifies where the dimensions of the smaller operand are aligned within the larger tensor.

Dimensions are counted from the **last dimension**, starting at zero:

| `dim` | Tensor dimension         |
| ----: | ------------------------ |
|   `0` | last dimension           |
|   `1` | second-to-last dimension |
|   `2` | third-to-last dimension  |
|   ... | ...                      |

The dimensions of the smaller operand must match the corresponding dimensions of the larger tensor.

Dimensions of the larger tensor outside the aligned range are broadcast.

## Example

Given:

```text
A.shape = (5, 2, 2, 3)
B.shape = (2, 3)
```

and:

```cpp
multiply(A, B, 0);
```

`B` is aligned with the last two dimensions of `A`:

```text
A: (5, 2, [2, 3])
B:       [2, 3]
```

The operation is equivalent to:

```text
result[a,b,c,d] = A[a,b,c,d] * B[c,d]
```

The values of `B[c,d]` are therefore reused across the first two dimensions of `A`.

## Vector broadcasting

For a vector:

```text
A.shape = (2, 3, 4)
v.shape = (4)
```

```cpp
multiply(A, v, 0);
```

aligns the vector with the last dimension:

```text
A: (2, 3, [4])
v:       [4]
```

which corresponds to:

```text
result[a,b,c] = A[a,b,c] * v[c]
```

To broadcast a vector along another dimension, `dim` can be changed accordingly.

## Scalars

A scalar does not require a broadcast dimension because it contains only one value and is applied to every element.

```cpp
multiply(A, scalar);
```

is equivalent to:

```text
result[i] = A[i] * scalar
```

## Operators

Operators perform strictly element-wise operations and do not perform explicit broadcasting.

```cpp
c = a * b;
```

requires compatible shapes.

For broadcasting, use the corresponding operation explicitly:

```cpp
c = multiply(a, b, dim);
```

This makes broadcasting visible at the call site.



