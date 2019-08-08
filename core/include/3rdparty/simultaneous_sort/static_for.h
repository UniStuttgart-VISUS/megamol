#pragma once

#include <utility>

/**
 Implementation based on
 https://codereview.stackexchange.com/questions/173564/implementation-of-static-for-to-iterate-over-elements-of-stdtuple-using-c17
*/

namespace
{
    template <std::size_t index>
    struct static_index
    {
        static constexpr std::size_t value = index;
    };

    template <class function_t, std::size_t... index_ts>
    constexpr void static_for_impl(function_t&& function, std::index_sequence<index_ts...>)
    {
        (function(static_index<index_ts>{}), ...);
    }
}

template <std::size_t n, class function_t>
constexpr void static_for(function_t&& function)
{
    static_for_impl(function, std::make_index_sequence<n>{});
}