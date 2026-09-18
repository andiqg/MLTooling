module;

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <span>


export module mlt.internal.compute.core.sizearray;

import mlt.core.error;

export namespace mlt::compute::core 
{
    inline constexpr size_t DEFAULT_DIM = 6;
    inline constexpr size_t SMALL_DIM = 4;
    inline constexpr size_t MID_DIM = 8;
    inline constexpr size_t LARGE_DIM = 16;

    template <size_t Size = DEFAULT_DIM>
    class SizeArray
    {
        private:
        std::array<size_t, Size> mData;
        size_t mLen;

        public:
        SizeArray() noexcept : mLen(Size) {}
        SizeArray(size_t len) noexcept : mLen(std::min(Size, len)) {}

        SizeArray(std::initializer_list<size_t> init) noexcept
            : mLen(std::min(Size, init.size()))
        {
            assert(init.size() <= Size);
            std::copy_n(init.begin(), mLen, mData.begin());
        }

        template<size_t N>
        SizeArray(const std::array<size_t, N>& init)
            : mLen(N)
        {
            static_assert(N <= Size, "Too many dimensions");
            std::copy(init.begin(), init.end(), mData.begin());
        }

        SizeArray(std::span<const size_t> shape)
            : mLen(std::min(Size, shape.size()))
        {
            assert(shape.size() <= Size);
            std::copy_n(shape.begin(), mLen, mData.begin());
        }


        ~SizeArray() = default;
        SizeArray(const SizeArray& other) noexcept = default; 
        SizeArray& operator=(const SizeArray& other) noexcept = default;
        SizeArray(SizeArray&& other) noexcept = default;
        SizeArray& operator=(SizeArray&& other) noexcept = default;

        size_t& operator[](size_t pos) noexcept
        {
            assert(pos < mLen);
            return mData[pos];
        }
        
        size_t operator[](size_t pos) const noexcept
        {
            assert(pos < mLen);
            return mData[pos];
        }

        size_t size() const noexcept { return mLen; }       

        size_t* getData() noexcept { return mData.data(); }
        const size_t* getData() const noexcept { return mData.data(); }

        std::span<const size_t> asSpan() const noexcept { return std::span(mData.data(), mLen); }
    };
    
    using DefaultSizeArray = SizeArray<>;
}
