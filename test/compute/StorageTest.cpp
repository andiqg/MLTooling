#include "../testdef.hpp"

#include <mlt/internal/compute/core/RefCount.hpp>

#include <cstddef>
#include <expected>
#include <span>

import mlt.core.error;
import mlt.internal.core.storage;

#define MLT_STORAGE_DEFAULT float

using namespace mlt::compute::core;

static constexpr size_t testSize = 10;

static void testStorageAlloc()
{
    std::expected<Ref<Storage>, mlt::core::MltError> result = Storage::alloc(testSize);

    Ref<Storage> s;
    if (result)
        s = result.value();
    else
        throwCustomMessage({result.error().msg.data()});

    assertEq(s->size, testSize, "Storage constructor allocated wrong amount of bytes.");
}

static void testStorageFrom()
{
    unsigned char raw[] = { 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x01 };
    std::span<std::byte> buf(reinterpret_cast<std::byte*>(raw), sizeof(raw));

    std::expected<Ref<Storage>, mlt::core::MltError> result = Storage::from(buf);
    
    Ref<Storage> s;
    
    if (result)
        s = result.value();
    else
        throwCustomMessage({result.error().msg.data()});

    assertEq(s->size, testSize, "Storage constructor allocated wrong amount of bytes.");

    std::byte* storageData = s->data;

    for (size_t i = 0; i < testSize; ++i)
        assertEq((int)storageData[i], (int)raw[i], "Storage constructor copied wrong.");
}

static void testStorageClone()
{
    unsigned char raw[] = { 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x01 };
    std::span<std::byte> buf(reinterpret_cast<std::byte*>(raw), sizeof(raw));

    mlt::core::Result<Ref<Storage>> result = Storage::from(buf);
    
    Ref<Storage> s;
    
    if (result)
        s = result.value();
    else
        throwCustomMessage({result.error().msg.data()});

    mlt::core::Result<Ref<Storage>> rCloned = s->clone();
    Ref<Storage> cloned;

    if (rCloned)
        cloned = rCloned.value();
    else
        throwCustomMessage({rCloned.error().msg.data()});

    std::byte* storageData = cloned->data;

    for (size_t i = 0; i < testSize; ++i)
        assertEq((int)storageData[i], (int)raw[i], "Storage cloned wrong.");
}

REGISTER_TEST(testStorageAlloc);
REGISTER_TEST(testStorageFrom);
REGISTER_TEST(testStorageClone);
