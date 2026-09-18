using MLTooling.Core;
using MLTooling.Core.Handles;

namespace MLTooling.Bindings;

using size_t = nuint;

internal static class MltWrapper
{
    public static TensorHandle TensorFrom(Shape shape)
    {
        MLTerror err = MltNative.mltTensorFrom(shape.ToNative(), out TensorHandle handle);
        MltError.ThrowIfError(err);
        handle.SetMemoryPressure(shape.ByteSize);

        return handle;
    }

    public static TensorHandle TensorFromBuffer(ReadOnlySpan<float> data, Shape shape)
    {
        MLTerror err = MltNative.mltTensorFromBuff(data, (size_t)data.Length, shape.ToNative(), out TensorHandle handle);
        MltError.ThrowIfError(err);
        handle.SetMemoryPressure(shape.ByteSize);

        return handle;
    }

    public static TensorHandle TensorCopy(TensorHandle src)
    {
        MLTerror err = MltNative.mltTensorCopy(src, out TensorHandle handle);
        MltError.ThrowIfError(err);

        return handle;
    }

    public static Shape TensorShape(TensorHandle handle)
    {
        MLTerror err = MltNative.mltTensorShape(handle, out MLTshape shape);
        MltError.ThrowIfError(err);

        return new Shape(shape);
    }

    public static float TensorGet(TensorHandle src, Shape shape)
    {
        MLTerror err = MltNative.mltTensorGet(src, shape.ToNative(), out float val);
        MltError.ThrowIfError(err);

        return val;
    }

    public static void TensorSet(TensorHandle src, Shape shape, float val)
    {
        MLTerror err = MltNative.mltTensorSet(src, shape.ToNative(), val);
        MltError.ThrowIfError(err);
    }

    public static TensorHandle TensorTranspose(TensorHandle src)
    {
        MLTerror err = MltNative.mltTensorTranspose(src, out TensorHandle handle);
        MltError.ThrowIfError(err);

        return handle;
    }

    public static TensorHandle TensorMatmul(TensorHandle a, TensorHandle b)
    {
        MLTerror err = MltNative.mltTensorMatmul(a, b, out TensorHandle handle);
        MltError.ThrowIfError(err);

        return handle;
    }
}
