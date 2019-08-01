#pragma once

#include <cstddef>

/// <summary>
/// Wrapper for values stored within a proxy
/// Implements all functionality that is required for sorting
/// </summary>
/// <template name="proxy_t">Proxy type</template>
template <typename proxy_t>
class value_wrapper
{
public:
    /// <summary>
    /// Create a wrapper for values stored within a proxy
    /// </summary>
    /// <param name="proxy">Proxy containing arrays</param>
    /// <param name="index">Index of the value stored in the proxy</param>
    value_wrapper(proxy_t* proxy, std::size_t index);

    /// <summary>
    /// Copy/move constructor/assignment
    /// </summary>
    value_wrapper(const value_wrapper& rhs);
    value_wrapper(value_wrapper&& rhs);

    value_wrapper& operator=(const value_wrapper& rhs);
    value_wrapper& operator=(value_wrapper&& rhs);

    value_wrapper& operator=(const typename proxy_t::value_type::type::value_type&);

    /// <summary>
    /// (De-)Referenciation
    /// </summary>
    typename proxy_t::value_type::type::value_type operator*();
    value_wrapper* operator->();

    operator typename proxy_t::value_type::type::value_type() const;

    /// <summary>
    /// Less operator for sorting operations
    /// </summary>
    bool operator<(const value_wrapper& rhs);

private:
    /// Index of the value stored in the proxy
    const std::size_t index;

    /// Proxy containing arrays
    proxy_t* proxy;
};

namespace std
{
    /// <summary>
    /// Swap implementation for wrapper
    /// </summary>
    template <typename proxy_t>
    void swap(value_wrapper<proxy_t> lhs, value_wrapper<proxy_t> rhs);
}
#include "value_wrapper.inl"
