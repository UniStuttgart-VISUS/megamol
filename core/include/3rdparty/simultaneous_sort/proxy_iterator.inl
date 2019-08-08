#include "proxy_iterator.h"

#include "value_wrapper.h"

#include <cstddef>
#include <iterator>

template <typename proxy_t>
inline proxy_iterator<proxy_t>::proxy_iterator() : proxy(nullptr), index(0) {}

template <typename proxy_t>
inline proxy_iterator<proxy_t>::proxy_iterator(proxy_t* proxy, const std::size_t index) : proxy(proxy), index(index) {}

template <typename proxy_t>
inline proxy_iterator<proxy_t>::proxy_iterator(const proxy_iterator& other) : proxy(other.proxy), index(other.index) {}

template <typename proxy_t>
inline proxy_iterator<proxy_t>::proxy_iterator(proxy_iterator&& other) : proxy(other.proxy), index(other.index) {}

template <typename proxy_t>
inline proxy_iterator<proxy_t>& proxy_iterator<proxy_t>::operator=(const proxy_iterator& other)
{
    this->proxy = other.proxy;
    this->index = other.index;
    return *this;
}

template <typename proxy_t>
inline proxy_iterator<proxy_t>& proxy_iterator<proxy_t>::operator=(proxy_iterator&& other)
{
    this->proxy = other.proxy;
    this->index = other.index;
    return *this;
}

template <typename proxy_t>
inline bool proxy_iterator<proxy_t>::operator==(const proxy_iterator& rhs) const
{
    return this->index == rhs.index;
}

template <typename proxy_t>
inline bool proxy_iterator<proxy_t>::operator!=(const proxy_iterator& rhs) const
{
    return this->index != rhs.index;
};

template <typename proxy_t>
inline bool proxy_iterator<proxy_t>::operator<(const proxy_iterator& rhs) const
{
    return this->index < rhs.index;
}

template <typename proxy_t>
inline bool proxy_iterator<proxy_t>::operator>(const proxy_iterator& rhs) const
{
    return this->index > rhs.index;
}

template <typename proxy_t>
inline bool proxy_iterator<proxy_t>::operator<=(const proxy_iterator& rhs) const
{
    return this->index <= rhs.index;
}

template <typename proxy_t>
inline bool proxy_iterator<proxy_t>::operator>=(const proxy_iterator& rhs) const
{
    return this->index >= rhs.index;
}

template <typename proxy_t>
inline value_wrapper<proxy_t> proxy_iterator<proxy_t>::operator*()
{
    return this->proxy->get(this->index);
}

template <typename proxy_t>
inline value_wrapper<proxy_t> proxy_iterator<proxy_t>::operator*() const
{
    return this->proxy->get(this->index);
}

template <typename proxy_t>
inline value_wrapper<proxy_t> proxy_iterator<proxy_t>::operator->()
{
    return this->proxy->get(this->index);
}

template <typename proxy_t>
inline value_wrapper<proxy_t> proxy_iterator<proxy_t>::operator->() const
{
    return this->proxy->get(this->index);
}

template <typename proxy_t>
inline value_wrapper<proxy_t> proxy_iterator<proxy_t>::operator[](const std::size_t i)
{
    return this->proxy->get(this->index + i);
}

template <typename proxy_t>
inline value_wrapper<proxy_t> proxy_iterator<proxy_t>::operator[](const std::size_t i) const
{
    return this->proxy->get(this->index + i);
}

template <typename proxy_t>
inline proxy_iterator<proxy_t>& proxy_iterator<proxy_t>::operator++()
{
    return *this += 1;
}

template <typename proxy_t>
inline proxy_iterator<proxy_t> proxy_iterator<proxy_t>::operator++(int)
{
    auto temp = *this;
    ++(*this);
    return temp;
}

template <typename proxy_t>
inline proxy_iterator<proxy_t>& proxy_iterator<proxy_t>::operator--()
{
    return *this -= 1;
}

template <typename proxy_t>
inline proxy_iterator<proxy_t> proxy_iterator<proxy_t>::operator--(int)
{
    auto temp = *this;
    --(*this);
    return temp;
}

template <typename proxy_t>
inline proxy_iterator<proxy_t> proxy_iterator<proxy_t>::operator+(const std::size_t n) const
{
    auto temp = *this;
    temp += n;
    return temp;
}

template <typename proxy_t>
inline proxy_iterator<proxy_t>& proxy_iterator<proxy_t>::operator+=(const std::size_t n)
{
    this->index += n;
    return *this;
}

template <typename proxy_t>
inline proxy_iterator<proxy_t> proxy_iterator<proxy_t>::operator-(const std::size_t n) const
{
    auto temp = *this;
    temp -= n;
    return temp;
}

template <typename proxy_t>
inline proxy_iterator<proxy_t>& proxy_iterator<proxy_t>::operator-=(const std::size_t n)
{
    this->index -= n;
    return *this;
}

template <typename proxy_t>
inline std::size_t proxy_iterator<proxy_t>::operator-(const proxy_iterator& rhs) const
{
    return rhs.get_index() > this->index ? rhs.get_index() - this->index : this->index - rhs.get_index();
}

template <typename proxy_t>
inline std::size_t proxy_iterator<proxy_t>::get_index() const
{
    return this->index;
}

template <typename proxy_t>
inline proxy_iterator<proxy_t> operator+(const std::size_t n, const proxy_iterator<proxy_t>& rhs)
{
    auto temp = rhs;
    temp += n;
    return temp;
}

template <typename proxy_t>
inline proxy_iterator<proxy_t> operator-(const std::size_t n, const proxy_iterator<proxy_t>& rhs)
{
    auto temp = rhs;
    temp -= n;
    return temp;
}