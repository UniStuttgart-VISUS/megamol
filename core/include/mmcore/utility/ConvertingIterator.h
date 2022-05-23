#pragma once

#include <iterator>

namespace megamol::core::utility {

// Alex' tactical nuke, i.e., iterator for mass-casting a whole iterable and get rid of tons of warnings.
template<typename U, typename T>
struct ConvertingIterator {
    using value_type = U;
    using reference = value_type;
    using pointer = value_type*;
    using iterator_category = std::random_access_iterator_tag;
    using difference_type = std::ptrdiff_t;

    // Construction
    ConvertingIterator(const T* current) : _current(current) {}

    // Access
    constexpr reference operator*() const noexcept {
        return static_cast<U>(*_current);
    }

    constexpr pointer operator->() const noexcept {
        return _current;
    }

    // Comparison
    constexpr bool operator==(const ConvertingIterator<U, T>& other) const noexcept {
        return other._current == _current;
    }

    constexpr bool operator!=(const ConvertingIterator<U, T>& other) const noexcept {
        return other._current != _current;
    }

    constexpr bool operator<(const ConvertingIterator<U, T>& other) const noexcept {
        return _current < other._current;
    }

    constexpr bool operator>(const ConvertingIterator<U, T>& other) const noexcept {
        return _current > other._current;
    }

    constexpr bool operator<=(const ConvertingIterator<U, T>& other) const noexcept {
        return _current <= other._current;
    }

    constexpr bool operator>=(const ConvertingIterator<U, T>& other) const noexcept {
        return _current >= other._current;
    }

    // Increment/Decrement
    ConvertingIterator<U, T> operator++(int) const {
        auto temp = ConvertingIterator<U, T>(_current);
        ++(*this);
        return temp;
    }

    ConvertingIterator<U, T>& operator++() {
        ++_current;
        return *this;
    }

    ConvertingIterator<U, T> operator--(int) const {
        auto temp = ConvertingIterator<U, T>(_current);
        --(*this);
        return temp;
    }

    ConvertingIterator<U, T>& operator--() {
        --_current;
        return *this;
    }

    // Add/Subtract
    constexpr ConvertingIterator<U, T>& operator+=(const difference_type other) noexcept {
        _current += other;
        return *this;
    }

    constexpr ConvertingIterator<U, T> operator+(const difference_type other) const noexcept {
        return ConvertingIterator<U, T>(_current + other);
    }

    constexpr ConvertingIterator<U, T>& operator-=(const difference_type other) noexcept {
        _current -= other;
        return *this;
    }

    constexpr ConvertingIterator<U, T> operator-(const difference_type other) const noexcept {
        return ConvertingIterator<U, T>(_current - other);
    }

    // Algebra
    constexpr difference_type operator-(const ConvertingIterator<U, T>& other) const noexcept {
        return std::distance(other._current, _current);
    }

    // Random access
    constexpr reference operator[](std::size_t index) const {
        return static_cast<U>(_current[index]);
    }

private:
    const T* _current;
};
} // namespace megamol::core::utility
