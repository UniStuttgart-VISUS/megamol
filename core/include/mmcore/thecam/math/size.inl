/*
 * the/math/size.inl
 *
 * Copyright (C) 2014 - 2016 TheLib Team (http://www.thelib.org/license)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * - Neither the name of TheLib, TheLib Team, nor the names of its
 *   contributors may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THELIB TEAM AS IS AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL THELIB TEAM BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 * TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


/*
 * megamol::core::thecam::math::size<V, D, T>::size
 */
template <class V, size_t D, class T> megamol::core::thecam::math::size<V, D, T>::size(void) {
    for (size_type i = 0; i < this->dimensions(); ++i) {
        traits_type::at(this->data, i) = static_cast<value_type>(0);
    }
}


/*
 * megamol::core::thecam::math::size<V, D, T>::size
 */
template <class V, size_t D, class T>
megamol::core::thecam::math::size<V, D, T>::size(std::initializer_list<value_type> il) {
    auto it = il.begin();
    for (size_type i = 0; i < this->dimensions(); ++i) {
        if (it != il.end()) {
            traits_type::at(this->data, i) = *it++;
        } else {
            traits_type::at(this->data, i) = static_cast<value_type>(0);
        }
    }
}


/*
 * megamol::core::thecam::math::size<V, D, T>::empty
 */
template <class V, size_t D, class T>
bool megamol::core::thecam::math::size<V, D, T>::empty(const value_type epsilon) const {
    for (size_type i = 0; i < this->dimensions(); ++i) {
        if (!is_equal(traits_type::at(this->data, i), static_cast<value_type>(0), epsilon)) {
            return false;
        }
    }
    return true;
}


/*
 * megamol::core::thecam::math::size<V, D, T>::equals
 */
template <class V, size_t D, class T>
template <class Vp, size_t Dp, class Tp>
bool megamol::core::thecam::math::size<V, D, T>::equals(const size<Vp, Dp, Tp>& rhs, const value_type epsilon) const {
    if (D == Dp) {
        for (size_type i = 0; i < this->dimensions(); ++i) {
            if (!is_equal(traits_type::at(this->data, i), rhs[i], epsilon)) {
                return false;
            }
        }
        return true;

    } else {
        // If the dimensions are not the same, the sizes cannot be equal.
        return false;
    }
}


/*
 * megamol::core::thecam::math::size<V, D, T>::volume
 */
template <class V, size_t D, class T>
typename megamol::core::thecam::math::size<V, D, T>::value_type megamol::core::thecam::math::size<V, D, T>::volume(
    void) const {
    auto retval = static_cast<value_type>(1);

    for (size_type i = 0; i < this->dimensions(); ++i) {
        retval *= traits_type::at(this->data, i);
    }

    return retval;
}


/*
 * megamol::core::thecam::math::size<V, D, T>::operator =
 */
template <class V, size_t D, class T>
megamol::core::thecam::math::size<V, D, T>& megamol::core::thecam::math::size<V, D, T>::operator=(const size& rhs) {
    if (this != std::addressof(rhs)) {
        traits_type::copy(this->data, rhs.data);
    }
    return *this;
}


/*
 * megamol::core::thecam::math::size<V, D, T>::operator =
 */
template <class V, size_t D, class T>
template <class Vp, class Tp>
megamol::core::thecam::math::size<V, D, T>& megamol::core::thecam::math::size<V, D, T>::operator=(
    const size<Vp, D, Tp>& rhs) {
    THE_ASSERT(static_cast<void*>(this) != static_cast<const void*>(std::addressof(rhs)));
    for (size_type i = 0; i < this->dimensions(); ++i) {
        traits_type::at(this->data, i) = static_cast<value_type>(rhs[i]);
    }
    return *this;
}
