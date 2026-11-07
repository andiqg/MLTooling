using Microsoft.Win32.SafeHandles;
using MLTooling.Bindings;


namespace MLTooling.Core.Handles;

abstract class NativeHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    public NativeHandle() : base(true)
    {}
}

internal sealed class TensorHandle : NativeHandle
{
    private const long MinPressureBytes = 4096;

    private long _pressureBytes;

    public TensorHandle()
    {}

    internal void SetMemoryPressure(long bytes)
    {
        if (bytes < MinPressureBytes || _pressureBytes != 0)
            return;

        _pressureBytes = bytes;
        GC.AddMemoryPressure(bytes);
    }

    protected override bool ReleaseHandle()
    {
        MltNative.mltTensorDestroy(handle);

        if (_pressureBytes > 0)
            GC.RemoveMemoryPressure(_pressureBytes);

        return true;
    }
}

