using MLTooling.Bindings;
using MLTooling.Core;
using MLTooling.Core.Handles;

namespace MLTooling.Compute;

public sealed class Tensor : IDisposable
{
    internal TensorHandle Handle { get; }
    public Shape Shape { get; }

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
        ArgumentNullException.ThrowIfNull(other);

        TensorHandle handle = MltWrapper.TensorMatmul(Handle, other.Handle);
        return FromNative(handle);
    }

    public Tensor ShareView() => FromNative(MltWrapper.TensorCopy(Handle));
    public Tensor Clone() => FromNative(MltWrapper.TensorClone(Handle));

    public void Dispose() => Handle.Dispose();

    private Tensor(Shape shape, TensorHandle handle)
    {
        Handle = handle;
        Shape = shape;
    }

    private static Tensor FromNative(TensorHandle handle)
        => new Tensor(MltWrapper.TensorShape(handle), handle);
}
