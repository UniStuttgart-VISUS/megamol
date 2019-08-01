#include "proxy.h"

#include "proxy_iterator.h"
#include "static_for.h"
#include "value_wrapper.h"

#include <stdexcept>
#include <tuple>
#include <utility>

template <typename t, typename... ts>
inline proxy<t, ts...>::proxy(std::tuple<t, ts...> arrays) : arrays(arrays), last_indices(std::make_pair<std::size_t, std::size_t>(0, 0))
{
    const std::size_t size = std::get<0>(this->arrays).get().size();

    static_for<std::tuple_size<std::tuple<t, ts...>>::value>([&](auto tuple_index)
    {
        if (std::get<tuple_index.value>(this->arrays).get().size() != size)
        {
            throw std::runtime_error("Arrays are of different size!");
        }
    });
}

template <typename t, typename... ts>
inline proxy_iterator<proxy<t, ts...>> proxy<t, ts...>::begin()
{
    return proxy_iterator<proxy<t, ts...>>(this);
}

template <typename t, typename... ts>
inline proxy_iterator<proxy<t, ts...>> proxy<t, ts...>::end()
{
    return proxy_iterator<proxy<t, ts...>>(this, std::get<0>(this->arrays).get().size());
}

template <typename t, typename... ts>
inline value_wrapper<proxy<t, ts...>> proxy<t, ts...>::get(std::size_t index)
{
    return value_wrapper<proxy<t, ts...>>(this, index);
}

template <typename t, typename... ts>
inline typename t::type::value_type& proxy<t, ts...>::get_value(std::size_t index)
{
    return std::get<0>(this->arrays).get()[index];
}

template <typename t, typename... ts>
inline void proxy<t, ts...>::swap(std::size_t lhs_index, std::size_t rhs_index)
{
    const auto min = std::min(lhs_index, rhs_index);
    const auto max = std::max(lhs_index, rhs_index);

    if (this->last_indices.first != min || this->last_indices.second != max)
    {
        static_for<std::tuple_size<std::tuple<t, ts...>>::value>([&](auto tuple_index)
        {
            std::swap(std::get<tuple_index.value>(this->arrays).get()[lhs_index], std::get<tuple_index.value>(this->arrays).get()[rhs_index]);
        });
    }

    this->last_indices.first = min;
    this->last_indices.second = max;
}
