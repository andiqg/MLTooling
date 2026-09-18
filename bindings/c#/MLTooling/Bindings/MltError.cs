namespace MLTooling.Bindings;

internal static class MltError
{
    public static void ThrowIfError(in MLTerror err)
    {
        if (err.IsOk)
            return;

        throw new MltException(err.Type, err.GetMessage());

    }
}

public sealed class MltException : Exception
{
    internal MLTErrorType ErrorType { get; }

    internal MltException(MLTErrorType errorType, string message)
        : base(message)
    {
        ErrorType = errorType;
    }
}
