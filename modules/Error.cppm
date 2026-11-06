module;

#include <array>
#include <cstddef>
#include <cstdio>
#include <expected>
#include <format>
#include <string_view>

export module mlt.core.error;
export namespace mlt::core 
{
    enum class MltErrorType
    {
        OutOfMemory,
        InvalidAlignment,
        OutOfBounds,
        TypeMismatch,
        RankMismatch,
        ShapeMismatch,
        InvalidArgument,
        InvalidBroadcastDim
    };

    constexpr std::string_view errorTemplate(MltErrorType type)
    {
        switch(type)
        {
            case MltErrorType::OutOfMemory:             return "Out of Memory.";
            case MltErrorType::InvalidAlignment:        return "Invalid alignment: {}";
            case MltErrorType::OutOfBounds:             return "Ouf of bounds: {}";
            case MltErrorType::TypeMismatch:            return "TypeMismatch: {}";
            case MltErrorType::RankMismatch:            return "RankMismatch: {}";
            case MltErrorType::ShapeMismatch:           return "ShapeMismatch: {}";
            case MltErrorType::InvalidArgument:         return "InvalidArgument: {}";
            case MltErrorType::InvalidBroadcastDim:     return "Invalid broadcast dimension {}";
        }
    }

    struct MltError
    {
        static constexpr size_t MAX_MSG_LEN = 50;
        
        MltErrorType type;
        std::array<char, MAX_MSG_LEN + 1> msg{};

        static MltError make(MltErrorType type, std::string_view detail = {}) noexcept
        {
            MltError err{type, {}};
            std::string_view tmpl = errorTemplate(type);
            
            int written;
            if (detail.empty())
                written = std::snprintf(err.msg.data(), err.msg.size(), "%.*s",
                            (int)tmpl.size(), tmpl.data());
            else
                written = std::snprintf(err.msg.data(), err.msg.size(), "%.*s: %.*s",
                            (int)tmpl.size(), tmpl.data(),
                            (int)detail.size(), detail.data());
            (void)written;
            return err;
        }

        static MltError makeTypeMismatch(std::string_view expected, std::string_view got)
        {
            return make(MltErrorType::TypeMismatch, std::format("Expected: {}, Got: {}", expected, got));
        }

        static MltError makeRankMismatch(const size_t expected, const size_t got)
        {
            return make(MltErrorType::RankMismatch, std::format("Expected: {}, Got: {}", expected, got));
        }

        static MltError makeOutOfBounds(const size_t max, const size_t got)
        {
            return make(MltErrorType::OutOfBounds, std::format("Max: {}, Got {}", max, got));
        }

        static MltError makeShapeMismatch()
        {
            return make(MltErrorType::ShapeMismatch);
        }
        
        static MltError makeInvalidArgument()
        {
            return make(MltErrorType::InvalidArgument);
        }

        static MltError makeInvalidBroadcastDim()
        {
            return make(MltErrorType::InvalidBroadcastDim);
        }
    };

    template <typename T>
    using Result = std::expected<T, MltError>;
}
