module;

#include <mlt/internal/compute/core/RefCount.hpp>
#include <cstddef>
#include <span>

export module mlt.internal.core.storage;

import mlt.core.error;

export namespace mlt::compute::core 
{
    constexpr size_t DEFAULT_ALIGNMENT = 64;

    struct Storage : RefCounted
    {
        std::byte* data;
        const size_t size;
        const size_t alignment;

        Storage(const Storage&) = delete;
        Storage& operator=(const Storage&) = delete;
        ~Storage();

        static mlt::core::Result<Ref<Storage>> alloc(const size_t bytes, const size_t alignment = DEFAULT_ALIGNMENT);
        static mlt::core::Result<Ref<Storage>> from(std::span<const std::byte> buf, const size_t alignment = DEFAULT_ALIGNMENT);
        mlt::core::Result<Ref<Storage>> clone() const noexcept;
        
    
        private:
        Storage(std::byte* data, size_t size, size_t alignment = DEFAULT_ALIGNMENT);
    };
}
