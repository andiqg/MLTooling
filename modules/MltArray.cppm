module;

#include <mlt/internal/compute/core/DType.hpp>
#include <mlt/internal/compute/core/RefCount.hpp>

#include <cassert>
#include <cstddef>
#include <span>
#include <type_traits>
#include <expected>

export module mlt.internal.compute.core.mltarray;

import mlt.core.error;
import mlt.internal.compute.core.sizearray;
import mlt.internal.core.storage;

export namespace mlt::compute::core 
{
    enum class DimType
    {
        COLUMN_MAJOR,
        ROW_MAJOR
    };


    struct MltArray
    {
        using default_dType = float;
        static constexpr DType DEFAULT_DTYPE = DType::FLOAT32;
        static constexpr DimType DEFAULT_DIM_TYPE = DimType::ROW_MAJOR;

        Ref<Storage> data;
        DefaultSizeArray strides;
        DefaultSizeArray shape;
        size_t offset;
        DType dType;
        DimType dimType;

        MltArray(const MltArray&) noexcept = default;
        MltArray& operator=(const MltArray&) noexcept = default;
        MltArray(MltArray&&) noexcept = default;
        MltArray& operator=(MltArray&&) noexcept = default;

        template <typename T = default_dType, typename... Indices>
        requires(std::is_convertible_v<Indices, size_t> &&...)
        T& operator[](Indices... idx)
        {
            assert(DTypeMapping<T>::value == dType);
            constexpr size_t idxLength = sizeof...(Indices);
            assert(shape.size() == idxLength);
            const size_t indices[] = { static_cast<size_t>(idx)... };
            size_t pos = offset;

            for (size_t i = 0; i < idxLength; ++i)
                pos += indices[i] * strides[i];

            return *(reinterpret_cast<T*>(data->data) + pos);
        }

        template <typename T = default_dType, typename... Indices>
        requires(std::is_convertible_v<Indices, size_t> &&...)
        const T operator[](Indices... idx) const
        {
            assert(DTypeMapping<T>::value == dType);
            constexpr size_t idxLength = sizeof...(Indices);
            assert(shape.size() == idxLength);
            const size_t indices[] = { static_cast<size_t>(idx)... };
            size_t pos = offset;

            for (size_t i = 0; i < idxLength; ++i)
                pos += indices[i] * strides[i];

            return *(reinterpret_cast<T*>(data->data) + pos);
        }
 
        // shape nxnxRowxCol
        static mlt::core::Result<MltArray> from(
            DefaultSizeArray shape,
            DType dType = DEFAULT_DTYPE,
            DimType dimType = DEFAULT_DIM_TYPE
        ) noexcept;

        template <typename T = default_dType>
        static std::expected<MltArray, mlt::core::MltError> from(
            std::span<const T> data,
            DefaultSizeArray shape,
            DimType dimType = DEFAULT_DIM_TYPE
        ) noexcept
        {
            const size_t count = shape.elemCount();
 
            if (count == 0)
                return std::unexpected(mlt::core::MltError::makeInvalidArgument());

            if (count != data.size())
                return std::unexpected(mlt::core::MltError::makeShapeMismatch());

            DefaultSizeArray strides = DefaultSizeArray(shape.size());

            MltArray::computeStrides(strides, shape, 1, dimType);

            std::span<const std::byte> bSpan = std::as_bytes(data);
            std::expected<Ref<Storage>, mlt::core::MltError> cStorage = Storage::from(bSpan);

            if (!cStorage)
                return std::unexpected(cStorage.error());

            Ref<Storage> storage = cStorage.value();

            return MltArray(
                storage,
                shape,
                strides,
                DTypeMapping<T>::value,
                dimType
            );
        }

        MltArray transpose() const noexcept;

        mlt::core::Result<MltArray> clone() const noexcept;

        template<typename T = default_dType, typename... Indices>
        requires (std::is_convertible_v<Indices, size_t>&& ...)
        mlt::core::Result<T*> at(Indices... idx) noexcept
        {
            if (DTypeMapping<T>::value != dType)
                return std::unexpected(mlt::core::MltError::makeTypeMismatch(
                            toString(DTypeMapping<T>::value),
                            toString(dType)
                ));

            constexpr size_t idxLength = sizeof...(Indices);
            const size_t rank = shape.size();

            if (idxLength != rank)
                return std::unexpected(mlt::core::MltError::makeRankMismatch(
                    rank,
                    idxLength
                ));

            size_t indices[] = { static_cast<size_t>(idx)... };
            size_t pos = offset;

            for (size_t i = 0; i < idxLength; ++i)
            {
                if (indices[i] >= shape[i])
                    return std::unexpected(mlt::core::MltError::makeOutOfBounds(
                        indices[i],
                        shape[i]
                    ));

                pos += indices[i] * strides[i];
            }
            
            return reinterpret_cast<T*>(data->data) + pos;
        }
        
        template <typename T = default_dType, typename... Indices>
        requires(std::is_convertible_v<Indices, size_t> && ...)
        mlt::core::Result<T> at(Indices... idx) const noexcept
        {
            if (DTypeMapping<T>::value != dType)
                return std::unexpected(mlt::core::MltError::makeTypeMismatch(
                            toString(DTypeMapping<T>::value),
                            toString(dType)
                ));

            constexpr size_t idxLength = sizeof...(Indices);
            const size_t rank = shape.size();
 
            if (idxLength != rank)
                return std::unexpected(mlt::core::MltError::makeRankMismatch(
                    rank,
                    idxLength
                ));

            const size_t indices[] = { static_cast<size_t>(idx)... };
            size_t pos = offset;

            for (size_t i = 0; i < idxLength; ++i)
            {
                if (indices[i] >= shape[i])
                    return std::unexpected(mlt::core::MltError::makeOutOfBounds(
                        indices[i],
                        shape[i]
                    ));
 
                pos += indices[i] * strides[i];
            }

            return *(reinterpret_cast<T*>(data->data) + pos);
        }

        template <typename T = default_dType>
        mlt::core::Result<T*> at(std::span<const size_t> indices) noexcept
        {
            if (DTypeMapping<T>::value != dType)
                return std::unexpected(mlt::core::MltError::makeTypeMismatch(
                            toString(DTypeMapping<T>::value),
                            toString(dType)
                ));

            const size_t rank = shape.size();
            const size_t idxLen = indices.size();

            if (idxLen != rank)
                return std::unexpected(mlt::core::MltError::makeRankMismatch(rank, idxLen));

            size_t pos = offset;

            for (size_t i = 0; i < idxLen; ++i)
            {
                if (indices[i] >= shape[i])
                    return std::unexpected(mlt::core::MltError::makeOutOfBounds(indices[i], shape[i]));

                pos += indices[i] * strides[i];
            }

           return reinterpret_cast<T*>(data->data) + pos;
        }

        template <typename T = default_dType>
        mlt::core::Result<T> at(std::span<const size_t> indices) const noexcept
        {
            if (DTypeMapping<T>::value != dType)
                return std::unexpected(mlt::core::MltError::makeTypeMismatch(
                            toString(DTypeMapping<T>::value),
                            toString(dType)
                ));

            const size_t rank = shape.size();
            const size_t idxLen = indices.size();

            if (idxLen != rank)
                return std::unexpected(mlt::core::MltError::makeRankMismatch(rank, idxLen));

            size_t pos = offset;

            for (size_t i = 0; i < idxLen; ++i)
            {
                if (indices[i] >= shape[i])
                    return std::unexpected(mlt::core::MltError::makeOutOfBounds(indices[i], shape[i]));

                pos += indices[i] * strides[i];
            }

            return *(reinterpret_cast<T*>(data->data) + pos);
        }

        bool isContiguous() const;
        bool isTransposed() const { return shape[shape.size() - 1] == 1; };
        bool isScalar() const { return shape.size() == 1 && shape[0] == 1; };
        bool isVector() const { return shape.size() == 1 && shape[0] > 1; }
        bool isMatrix() const { return shape.size() == 2; }
        
        template <typename T = default_dType>
        mlt::core::Result<const T*> asPtr() const
        {
            DType currentType = DTypeMapping<T>::value;

            if (currentType != dType)
                return std::unexpected(mlt::core::MltError::makeTypeMismatch(toString(currentType), toString(dType)));

            return reinterpret_cast<const T*>(data->data);
        };
        
        template <typename T = default_dType>
        mlt::core::Result<T*> asPtr() 
        {
            DType currentType = DTypeMapping<T>::value;
            
            if (currentType != dType)
                return std::unexpected(mlt::core::MltError::makeTypeMismatch(toString(currentType), toString(dType)));
            
            return reinterpret_cast<T*>(data->data);
        }

        template <typename T>
        const T* asPtrUnchecked() const {  return reinterpret_cast<const T*>(data->data); }
        
        template <typename T>
        T* asPtrUnchecked() { return reinterpret_cast<T*>(data->data); }


        private:
        MltArray(
            Ref<Storage> storage,
            DefaultSizeArray shape,
            DefaultSizeArray strides,
            DType dType,
            DimType dimType
        ) noexcept;

        static void computeStrides(
            DefaultSizeArray& stride,
            const DefaultSizeArray& shape,
            const size_t startStride,
            const DimType type
        ) noexcept
        {
            switch (type)
            {
                case DimType::ROW_MAJOR:
                {
                    const size_t len = shape.size();
                    size_t currentStride = startStride;

                    for (size_t i = len; i-- > 0;)
                    {
                        stride[i] = currentStride;
                        currentStride *= shape[i];
                    }

                    break;
                }
                case DimType::COLUMN_MAJOR:
                {
                    const size_t len = shape.size();
                    size_t currentStride = startStride;

                    for (size_t i = 0; i < len; ++i)
                    {
                        stride[i] = currentStride;
                        currentStride *= shape[i];
                    }

                    break;
                }
            }
        }
    };
}
