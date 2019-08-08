#pragma once

#include "proxy_iterator.h"
#include "value_wrapper.h"

#include <stdexcept>
#include <tuple>
#include <utility>

/// <summary>
/// Proxy for storing arrays and performing swaps on them simultaneously
/// </summary>
/// <template name="t">Type of first array, which is used for sorting</template>
/// <template name="ts">Types of further arrays, which are reordered accordingly</template>
template <typename t, typename... ts>
class proxy
{
public:
    using value_type = t;

    /// <summary>
    /// Create a proxy object from a tuple of arrays
    /// </summary>
    /// <param name="arrays">Arrays used for sorting</param>
    proxy(std::tuple<t, ts...> arrays);

    /// <summary>
    /// Return iterator pointing at the first, or past the last, element, respectively
    /// </summary>
    /// <returns>Iterator pointing at the first, or past the last, element, respectively</returns>
    proxy_iterator<proxy<t, ts...>> begin();
    proxy_iterator<proxy<t, ts...>> end();

    /// <summary>
    /// Return wrapper for values at the given index
    /// </summary>
    /// <param name="index">Index of values within the arrays</param>
    /// <returns>Value wrapper</returns>
    value_wrapper<proxy<t, ts...>> get(std::size_t index);

    /// <summary>
    /// Return the value stored in the first array at given index
    /// </summary>
    /// <param name="index">Index of the value to get from the first array</param>
    /// <returns>Value at the given position from the first array</returns>
    typename t::type::value_type& get_value(std::size_t index);

    /// <summary>
    /// Swap two values, as indicated by the given indices, in all arrays alike
    /// </summary>
    /// <param name="lhs_index">Index of the first value</param>
    /// <param name="rhs_index">Index of the other value</param>
    void swap(std::size_t lhs_index, std::size_t rhs_index);

private:
    /// Stored arrays
    std::tuple<t, ts...> arrays;

    /// Last indices needed for validation of swaps
    std::pair<std::size_t, std::size_t> last_indices;
};

#include "proxy.inl"