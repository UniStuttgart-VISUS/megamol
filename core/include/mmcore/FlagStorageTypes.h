#pragma once

#include <type_traits>
#include <vector>

template<typename Enum>
struct EnableBitMaskOperators {
    static const bool enable = false;
};

template<typename Enum>
constexpr typename std::enable_if<EnableBitMaskOperators<Enum>::enable, Enum>::type operator|(Enum lhs, Enum rhs) {
    using underlying = typename std::underlying_type<Enum>::type;
    return static_cast<Enum>(static_cast<underlying>(lhs) | static_cast<underlying>(rhs));
}

template<typename Enum>
constexpr typename std::enable_if<EnableBitMaskOperators<Enum>::enable, Enum>::type operator&(Enum lhs, Enum rhs) {
    using underlying = typename std::underlying_type<Enum>::type;
    return static_cast<Enum>(static_cast<underlying>(lhs) & static_cast<underlying>(rhs));
}

template<typename Enum>
constexpr typename std::enable_if<EnableBitMaskOperators<Enum>::enable, Enum>::type operator^(Enum lhs, Enum rhs) {
    using underlying = typename std::underlying_type<Enum>::type;
    return static_cast<Enum>(static_cast<underlying>(lhs) ^ static_cast<underlying>(rhs));
}

// not sure I want that
//template<typename Enum>
//typename std::enable_if<EnableBitMaskOperators<Enum>::enable, Enum>::type operator~(Enum rhs) {
//    using underlying = typename std::underlying_type<Enum>::type;
//    return static_cast<Enum>(static_cast<underlying>(lhs) | static_cast<underlying>(rhs));
//}

// do we want the assignment ops?

namespace megamol {
namespace core {

/**
 * Class holding a buffer of uints which contain flags that say something
 * about a synchronized other piece of data (index equality).
 * Can be used for storing selection etc.
 */
class FlagStorageTypes {
public:
    using index_type = int32_t;
    using index_vector = std::vector<index_type>;
    using flag_item_type = uint32_t;
    using flag_version_type = uint32_t;
    using flag_vector_type = std::vector<flag_item_type>;

    enum class flag_bits : flag_item_type {
        ENABLED = 1 << 0,
        FILTERED = 1 << 1,
        SELECTED = 1 << 2,
    };

    template<typename E>
    static constexpr auto to_integral(const E e) -> typename std::underlying_type<E>::type {
        return static_cast<typename std::underlying_type<E>::type>(e);
    }
};

} // namespace core
} // namespace megamol

template<>
struct EnableBitMaskOperators<megamol::core::FlagStorageTypes::flag_bits> {
    static const bool enable = true;
};
