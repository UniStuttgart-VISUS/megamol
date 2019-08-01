#include "value_wrapper.h"

#include <algorithm>
#include <cstddef>

template <typename proxy_t>
inline value_wrapper<proxy_t>::value_wrapper(proxy_t* proxy, std::size_t index) : proxy(proxy), index(index) { }

template <typename proxy_t>
inline value_wrapper<proxy_t>::value_wrapper(const value_wrapper& rhs) : proxy(rhs.proxy), index(rhs.index) { }

template <typename proxy_t>
inline value_wrapper<proxy_t>::value_wrapper(value_wrapper&& rhs) : proxy(rhs.proxy), index(rhs.index) { }

template <typename proxy_t>
inline value_wrapper<proxy_t>& value_wrapper<proxy_t>::operator=(const value_wrapper& rhs)
{
    if (index != rhs.index)
    {
        this->proxy->swap(index, rhs.index);
    }

    return *this;
}

template <typename proxy_t>
inline value_wrapper<proxy_t>& value_wrapper<proxy_t>::operator=(value_wrapper&& rhs)
{
    this->operator=(static_cast<value_wrapper&>(rhs));

    return *this;
}

template <typename proxy_t>
inline value_wrapper<proxy_t>& value_wrapper<proxy_t>::operator=(const typename proxy_t::value_type::type::value_type&)
{
    return *this;
}

template <typename proxy_t>
inline typename proxy_t::value_type::type::value_type value_wrapper<proxy_t>::operator*()
{
    return this->proxy->get_value(this->index);
}

template <typename proxy_t>
inline value_wrapper<proxy_t>* value_wrapper<proxy_t>::operator->()
{
    return this;
}

template <typename proxy_t>
inline value_wrapper<proxy_t>::operator typename proxy_t::value_type::type::value_type() const
{
    return this->proxy->get_value(this->index);
}

template <typename proxy_t>
inline bool value_wrapper<proxy_t>::operator<(const value_wrapper& other)
{
    return this->proxy->get_value(this->index) < other.proxy->get_value(other.index);
}

namespace std
{
    template <typename proxy_t>
    inline void swap(value_wrapper<proxy_t> lhs, value_wrapper<proxy_t> rhs)
    {
        lhs = rhs;
    }
}
