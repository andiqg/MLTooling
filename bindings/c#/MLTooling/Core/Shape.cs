using MLTooling.Bindings;

namespace MLTooling.Core;

using size_t = nuint;

public readonly struct Shape : IEquatable<Shape>
{
    public const int MaxDims = MltNative.MaxDims;

    private readonly ulong[] _dims;

    public Shape(params ulong[] dims)
    {
        ArgumentNullException.ThrowIfNull(dims);

        if (dims.Length is 0 or > MaxDims)
            throw new ArgumentException($"Shape braucht 1 bis {MaxDims} Dimensionen.", nameof(dims));

        foreach (int d in dims)
            if (d < 0)
                throw new ArgumentOutOfRangeException(nameof(dims), "Dimensionen müssen > 0 sein.");

        _dims = (ulong[])dims.Clone();
    }

    internal Shape(MLTshape shape)
    {
        _dims = new ulong[shape.Size];

        for (int i = 0; i < _dims.Length; i++)
            _dims[i] = shape.Dims[i];
    }

    public int Rank => _dims.Length;
    public ReadOnlySpan<ulong> Dims => _dims;

    public long ElementCount
    {
        get
        {
            long total = 1;
            foreach (int d in _dims) total *= d;
            
            return total;
        }
    }

    public long ByteSize => ElementCount * sizeof(float);

    internal MLTshape ToNative()
    {
        MLTshape native = default;

        for (int i = 0; i < _dims.Length; i++)
            native.Dims[i] = (size_t)_dims[i];
        native.Size = (size_t)_dims.Length;

        return native;
    }

    public bool Equals(Shape other) => Dims.SequenceEqual(other.Dims);
    public override bool Equals(object? obj) => obj is Shape s && Equals(s);

    public override int GetHashCode()
    {
        HashCode hc = new();
        foreach (int d in _dims) hc.Add(d);

        return hc.ToHashCode();
    }

    public override string ToString() => $"({string.Join(", ", _dims)})";
} 
