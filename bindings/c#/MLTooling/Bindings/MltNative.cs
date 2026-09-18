using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using System.Text;

using MLTooling.Core.Handles;

namespace MLTooling.Bindings;

using size_t = nuint;
using ptr = nint;

internal enum MLTErrorType : int
{
    Ok = 0,
    NullPointer,
    OutOfMemory,
    InvalidAlignment,
    OutOfBounds,
    TypeMismatch,
    RankMismatch,
    ShapeMismatch,
    InvalidArgument,
    InternalError,
}

[InlineArray(MltNative.MaxDims)]
internal struct MLTDims
{
    private size_t _element0;
}

[StructLayout(LayoutKind.Sequential)]
internal struct MLTshape
{
    public MLTDims Dims;
    public size_t Size;
}

[InlineArray(MltNative.MaxErrorMsgLen + 1)]
internal struct MltErrorMessage
{
    private byte _element0;
}

[StructLayout(LayoutKind.Sequential)]
internal struct MLTerror
{
    public MLTErrorType Type;
    public MltErrorMessage Message;

    internal readonly bool IsOk => Type == MLTErrorType.Ok;

    internal readonly string GetMessage()
    {
        ReadOnlySpan<byte> raw = Message;
        int nullIndex = raw.IndexOf((byte)0);
        var bytes = nullIndex >= 0 ? raw[..nullIndex] : raw;
        
        return Encoding.UTF8.GetString(bytes);
    }
}

internal static partial class MltNative
{
    internal const int MaxErrorMsgLen = 50;
    internal const int MaxDims = 6;

    private const string DllName = "MLToolingC";
    private const CallingConvention _conv = CallingConvention.Cdecl;

    [LibraryImport(DllName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallingConvention)])]
    internal static partial MLTerror mltTensorFrom(MLTshape shape, out TensorHandle tensor);

    [LibraryImport(DllName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallingConvention)])]
    internal static partial MLTerror mltTensorFromBuff(ReadOnlySpan<float> buf, size_t len, MLTshape shape, out TensorHandle tensor);


    [LibraryImport(DllName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallingConvention)])]
    internal static partial void mltTensorDestroy(ptr tensor);

    [LibraryImport(DllName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallingConvention)])]
    internal static partial MLTerror mltTensorCopy(TensorHandle src, out TensorHandle tensor);

    [LibraryImport(DllName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallingConvention)])]
    internal static partial MLTerror mltTensorShape(TensorHandle tensor, out MLTshape shape);

    [LibraryImport(DllName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallingConvention)])]
    internal static partial MLTerror mltTensorGet(TensorHandle tensor, MLTshape shape, out float val);

    [LibraryImport(DllName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallingConvention)])]
    internal static partial MLTerror mltTensorSet(TensorHandle tensor, MLTshape shape, float val);

    [LibraryImport(DllName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallingConvention)])]
    internal static partial MLTerror mltTensorTranspose(TensorHandle src, out TensorHandle tensor);

    [LibraryImport(DllName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallingConvention)])]
    internal static partial MLTerror mltTensorMatmul(TensorHandle a, TensorHandle b, out TensorHandle c);
}
