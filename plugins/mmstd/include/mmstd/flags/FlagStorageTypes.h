/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <cstdint>
#include <vector>

#include "EnumBitMaskOperators.h"

namespace megamol::core {

class FlagStorageTypes {
public:
    using index_type = int32_t;
    using index_vector = std::vector<index_type>;
    using flag_item_type = uint32_t;
    using flag_version_type = uint32_t;
    using flag_vector_type = std::vector<flag_item_type>;

    enum class flag_bits : flag_item_type {
        ENABLED = 1 << 0,  // the item is active, can be selected, clicked, manipulated
        FILTERED = 1 << 1, // the item is filtered and should not be visible in any renderer,
                           // at most as a ghost or a silhouette
        SELECTED = 1 << 2, // the item is selected and should be highlighted in all renderers
    };

    template<typename E>
    static constexpr auto to_integral(const E e) -> typename std::underlying_type<E>::type {
        return static_cast<typename std::underlying_type<E>::type>(e);
    }
};

} // namespace megamol::core

template<>
struct EnableBitMaskOperators<megamol::core::FlagStorageTypes::flag_bits> {
    static const bool enable = true;
};
