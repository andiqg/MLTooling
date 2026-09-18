using System.Runtime.InteropServices;

using MLTooling.Bindings;
using MLTooling.Core;
using MLTooling.Core.Handles;

namespace MLTooling.Compute;

public sealed class Tensor : IDisposable
{
    internal TensorHandle Handle { get; }
    public Shape Shape { get; }

    public SafeHandle DangerousHandle => Handle;

    public Tensor(Shape shape)
    {
        Handle = MltWrapper.TensorFrom(shape);
        Shape = shape;
    }

    public Tensor(ReadOnlySpan<float> data, Shape shape)
    {
        if (data.Length != shape.ElementCount)
        {
            throw new ArgumentException(
                    $"Datenlänge ({data.Length}) passt nicht zur Shape {shape} ({shape.ElementCount} Elemente)",
                    nameof(data)
            );
        }

        Handle = MltWrapper.TensorFromBuffer(data, shape);
        Shape = shape;
    }

    public float this[params ulong[] indices]
    {
        get => MltWrapper.TensorGet(Handle, new Shape(indices));
        set => MltWrapper.TensorSet(Handle, new Shape(indices), value);
    }

    public Tensor Matmul(Tensor other)
    {
        TensorHandle handle = MltWrapper.TensorMatmul(Handle, other.Handle);
        Shape shape = MltWrapper.TensorShape(Handle);

        return new Tensor(shape, handle);
    }

    public Tensor ShareView()
    {
        TensorHandle handle = MltWrapper.TensorCopy(Handle);

        return new Tensor(Shape, handle);
    }

    public void Dispose() => Handle.Dispose();

    private Tensor(Shape shape, TensorHandle handle)
    {
        Handle = handle;
        Shape = shape;
    }
}
