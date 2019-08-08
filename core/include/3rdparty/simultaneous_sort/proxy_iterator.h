#pragma once

#include "value_wrapper.h"

#include <cstddef>
#include <iterator>

/// <summary>
/// Iterator for proxy, dereferencing to a wrapped value
/// </summary>
/// <template name="proxy_t">Proxy type</template>
template <typename proxy_t>
class proxy_iterator
{
public:
    /// Typedefs for iterator
    using iterator_category = std::random_access_iterator_tag;
    using value_type = typename proxy_t::value_type::type::value_type;
    using difference_type = std::size_t;
    using pointer = value_wrapper<proxy_t>;
    using reference = value_wrapper<proxy_t>;

    /// <summary>
    /// Create proxy iterator from proxy and initial index
    /// </summary>
    /// <param name="proxy">Proxy containing arrays</param>
    /// <param name="index">Index for pointing at the correct values</param>
    proxy_iterator();
    proxy_iterator(proxy_t* proxy, std::size_t index = 0);

    /// <summary>
    /// Copy/move constructor/assignment
    /// </summary>
    proxy_iterator(const proxy_iterator& other);
    proxy_iterator(proxy_iterator&& other);

    proxy_iterator& operator=(const proxy_iterator& other);
    proxy_iterator& operator=(proxy_iterator&& other);

    /// <summary>
    /// Comparison operators
    /// </summary>
    bool operator==(const proxy_iterator& rhs) const;
    bool operator!=(const proxy_iterator& rhs) const;
    bool operator<(const proxy_iterator& rhs) const;
    bool operator>(const proxy_iterator& rhs) const;
    bool operator<=(const proxy_iterator& rhs) const;
    bool operator>=(const proxy_iterator& rhs) const;

    /// <summary>
    /// Dereferencing to wrapped values
    /// </summary>
    /// <returns>Wrapped values</returns>
    value_wrapper<proxy_t> operator*();
    value_wrapper<proxy_t> operator*() const;

    value_wrapper<proxy_t> operator->();
    value_wrapper<proxy_t> operator->() const;

    value_wrapper<proxy_t> operator[](std::size_t i);
    value_wrapper<proxy_t> operator[](std::size_t i) const;

    /// <summary>
    /// Arithmetic operations for providing random access
    /// </summary>
    proxy_iterator& operator++();
    proxy_iterator operator++(int);
    proxy_iterator& operator--();
    proxy_iterator operator--(int);
    proxy_iterator operator+(std::size_t n) const;
    proxy_iterator& operator+=(std::size_t n);
    proxy_iterator operator-(std::size_t n) const;
    proxy_iterator& operator-=(std::size_t n);

    /// <summary>
    /// Arithmetic operation for providing distance between iterators
    /// </summary>
    std::size_t operator-(const proxy_iterator& rhs) const;

    /// <summary>
    /// Get index of this iterator
    /// </summary>
    /// <returns></returns>
    std::size_t get_index() const;

private:
    /// Index pointing at the values stored within the proxy
    std::size_t index;

    /// Proxy containing value arrays
    proxy_t* proxy;
};

/// <summary>
/// Arithmetic operations for providing random access
/// </summary>
template <typename proxy_t>
proxy_iterator<proxy_t> operator+(std::size_t n, const proxy_iterator<proxy_t>& rhs);

template <typename proxy_t>
proxy_iterator<proxy_t> operator-(std::size_t n, const proxy_iterator<proxy_t>& rhs);

#include "proxy_iterator.inl"