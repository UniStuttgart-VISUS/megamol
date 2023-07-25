/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <iterator>

namespace megamol::core::utility {

// Alex' tactical nuke, i.e., iterator for mass-casting a whole iterable and get rid of tons of warnings.
template<typename To, typename From>
struct ConvertingIterator {
    using value_type = To;
    using reference = const value_type;
    using pointer = typename std::conditional<std::is_same<To, From>::value, value_type*, const void*>::type;
    using iterator_category = std::random_access_iterator_tag;
    using difference_type = std::ptrdiff_t;

    // Construction
    ConvertingIterator(const From* current) : _current(current) {}

    // Access
    constexpr reference operator*() const noexcept {
        return static_cast<To>(*_current);
    }

    template<typename U_ = To, typename T_ = From>
    constexpr typename std::enable_if<std::is_same<U_, T_>::value, pointer>::type operator->() const noexcept {
        return _current;
    }

    template<typename U_ = To, typename T_ = From>
    constexpr typename std::enable_if<!std::is_same<U_, T_>::value>::type operator->() const noexcept {
        static_assert(std::is_same<U_, T_>::value, "Pointer access not possible.");
    }

    // Comparison
    constexpr bool operator==(const ConvertingIterator<To, From>& other) const noexcept {
        return other._current == _current;
    }

    constexpr bool operator!=(const ConvertingIterator<To, From>& other) const noexcept {
        return other._current != _current;
    }

    constexpr bool operator<(const ConvertingIterator<To, From>& other) const noexcept {
        return _current < other._current;
    }

    constexpr bool operator>(const ConvertingIterator<To, From>& other) const noexcept {
        return _current > other._current;
    }

    constexpr bool operator<=(const ConvertingIterator<To, From>& other) const noexcept {
        return _current <= other._current;
    }

    constexpr bool operator>=(const ConvertingIterator<To, From>& other) const noexcept {
        return _current >= other._current;
    }

    // Increment/Decrement
    ConvertingIterator<To, From> operator++(int) const {
        auto temp = ConvertingIterator<To, From>(_current);
        ++(*this);
        return temp;
    }

    ConvertingIterator<To, From>& operator++() {
        ++_current;
        return *this;
    }

    ConvertingIterator<To, From> operator--(int) const {
        auto temp = ConvertingIterator<To, From>(_current);
        --(*this);
        return temp;
    }

    ConvertingIterator<To, From>& operator--() {
        --_current;
        return *this;
    }

    // Add/Subtract
    constexpr ConvertingIterator<To, From>& operator+=(const difference_type other) noexcept {
        _current += other;
        return *this;
    }

    constexpr ConvertingIterator<To, From> operator+(const difference_type other) const noexcept {
        return ConvertingIterator<To, From>(_current + other);
    }

    constexpr ConvertingIterator<To, From>& operator-=(const difference_type other) noexcept {
        _current -= other;
        return *this;
    }

    constexpr ConvertingIterator<To, From> operator-(const difference_type other) const noexcept {
        return ConvertingIterator<To, From>(_current - other);
    }

    constexpr difference_type operator-(const ConvertingIterator<To, From>& other) const noexcept {
        return std::distance(other._current, _current);
    }

    // Random access
    constexpr reference operator[](std::size_t index) const {
        return static_cast<To>(_current[index]);
    }

private:
    const From* _current;
};
} // namespace megamol::core::utility
