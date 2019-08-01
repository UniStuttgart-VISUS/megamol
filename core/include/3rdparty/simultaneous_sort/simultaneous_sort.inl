#include "simultaneous_sort.h"

#include "proxy.h"
#include "static_for.h"

#include <algorithm>
#include <functional>
#include <tuple>

template <typename first_array_t, typename... array_ts>
void sort(first_array_t& first_array, array_ts&... arrays)
{
    sort_with([](const typename first_array_t::value_type& lhs, const typename first_array_t::value_type& rhs)
    {
        return lhs < rhs;
    }
    , first_array, arrays...);
}

template <typename predicate_t, typename... array_ts>
void sort_with(predicate_t predicate, array_ts&... arrays)
{
    proxy<std::reference_wrapper<array_ts>...> sort_proxy(std::make_tuple(std::ref(arrays)...));

    std::sort(sort_proxy.begin(), sort_proxy.end(), predicate);
}
