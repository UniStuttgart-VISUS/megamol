/*
 * thecam\math\vector.inl
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
 * megamol::core::thecam::math::vector<V, D, T>::vector
 */
template <class V, size_t D, class T>
megamol::core::thecam::math::vector<V, D, T>::vector(std::initializer_list<value_type> il) {
    auto it = il.begin();
    for (size_type i = 0; i < this->size(); ++i) {
        if (it != il.end()) {
            traits_type::at(this->data, i) = *it++;
        } else {
            traits_type::at(this->data, i) = static_cast<value_type>(0);
        }
    }
}


/*
 * megamol::core::thecam::math::vector<V, D, T>::vector
 */
template <class V, size_t D, class T>
template <class Vp, size_t Dp, class Tp>
megamol::core::thecam::math::vector<V, D, T>::vector(const vector<Vp, Dp, Tp>& rhs, const value_type value) {
    size_type i = 0;
    for (; i < (std::min)(this->size(), rhs.size()); ++i) {
        traits_type::at(this->data, i) = static_cast<value_type>(rhs[i]);
    }
    for (; i < this->size(); ++i) {
        traits_type::at(this->data, i) = value;
    }
}


/*
 * megamol::core::thecam::math::vector<V, D, T>::empty
 */
template <class V, size_t D, class T>
bool megamol::core::thecam::math::vector<V, D, T>::empty(const value_type epsilon) const {
    for (size_type i = 0; i < this->size(); ++i) {
        if (!is_equal(traits_type::at(this->data, i), static_cast<value_type>(0), epsilon)) {
            return false;
        }
    }
    return true;
}


/*
 * megamol::core::thecam::math::vector<V, D, T>::equals
 */
template <class V, size_t D, class T>
template <class Vp, size_t Dp, class Tp>
bool megamol::core::thecam::math::vector<V, D, T>::equals(
    const vector<Vp, Dp, Tp>& rhs, const value_type epsilon) const {
    if (D == Dp) {
        for (size_type i = 0; i < this->size(); ++i) {
            if (!is_equal(traits_type::at(this->data, i), rhs[i], epsilon)) {
                return false;
            }
        }
        return true;

    } else {
        // If the dimensions are not the same, the vectors cannot be equal.
        return false;
    }
}


/*
 * megamol::core::thecam::math::vector<V, D, T>::operator =
 */
template <class V, size_t D, class T>
megamol::core::thecam::math::vector<V, D, T>& megamol::core::thecam::math::vector<V, D, T>::operator=(
    const vector& rhs) {
    if (this != std::addressof(rhs)) {
        traits_type::copy(this->data, rhs.data);
    }
    return *this;
}


/*
 * megamol::core::thecam::math::vector<V, D, T>::operator =
 */
template <class V, size_t D, class T>
template <class Vp, class Tp>
megamol::core::thecam::math::vector<V, D, T>& megamol::core::thecam::math::vector<V, D, T>::operator=(
    const vector<Vp, D, Tp>& rhs) {
    THE_ASSERT(static_cast<void*>(this) != static_cast<const void*>(std::addressof(rhs)));
    for (size_type i = 0; i < this->size(); ++i) {
        traits_type::at(this->data, i) = static_cast<value_type>(rhs[i]);
    }
    return *this;
}


///*
// * megamol::core::thecam::math::vector<V, D, T>::alloc
// */
// template<class V, size_t D, class T>
// typename megamol::core::thecam::math::vector<V, D, T>::traits_type::allocator_type
// megamol::core::thecam::math::vector<V, D, T>::alloc;


/*
 * megamol::core::thecam::math::cross
 */
template <class V, class T>
megamol::core::thecam::math::vector<V, 3, T> megamol::core::thecam::math::cross(
    const vector<V, 3, T>& lhs, const vector<V, 3, T>& rhs) {
    thecam::math::vector<V, 3, T> retval(lhs.y() * rhs.z() - lhs.z() * rhs.y(), lhs.z() * rhs.x() - lhs.x() * rhs.z(),
        lhs.x() * rhs.y() - lhs.y() * rhs.x());
    return std::move(retval);
}


/*
 * megamol::core::thecam::math::cross
 */
template <class V, class T>
megamol::core::thecam::math::vector<V, 4, T> megamol::core::thecam::math::cross(
    const vector<V, 4, T>& lhs, const vector<V, 4, T>& rhs) {
    typedef typename T::value_type value_type;
    thecam::math::vector<V, 4, T> retval(lhs.y() * rhs.z() - lhs.z() * rhs.y(), lhs.z() * rhs.x() - lhs.x() * rhs.z(),
        lhs.x() * rhs.y() - lhs.y() * rhs.x(), static_cast<value_type>(0));
    return std::move(retval);
}


/*
 * megamol::core::thecam::math::dot
 */
template <class V, size_t D, class T>
typename T::value_type megamol::core::thecam::math::dot(const vector<V, D, T>& lhs, const vector<V, D, T>& rhs) {
    typedef typename T::size_type size_type;
    typedef typename T::value_type value_type;
    value_type retval = static_cast<value_type>(0);

    for (size_type i = 0; i < lhs.size(); ++i) {
        retval += lhs[i] * rhs[i];
    }

    return retval;
}


/*
 * megamol::core::thecam::math::normalise
 */
template <class V, size_t D, class T>
megamol::core::thecam::math::vector<V, D, T> megamol::core::thecam::math::normalise(const vector<V, D, T>& vec) {
    typedef typename T::size_type size_type;
    typedef typename T::value_type value_type;

    auto l = square_length(vec);

    if (l != static_cast<value_type>(0)) {
        l = sqrt(l);
        vector<V, D, T> retval(megamol::core::thecam::utility::do_not_initialise);
        for (size_type i = 0; i < vec.size(); ++i) {
            retval[i] = vec[i] / l;
        }
        return std::move(retval);

    } else {
        vector<V, D, T> retval;
        THE_ASSERT(retval.empty());
        return retval;
    }
}


/*
 * megamol::core::thecam::math::set_empty
 */
template <class V, size_t D, class T>
megamol::core::thecam::math::vector<V, D, T>& megamol::core::thecam::math::set_empty(vector<V, D, T>& vec) {
    typedef typename T::size_type size_type;
    typedef typename T::value_type value_type;

    for (size_type i = 0; i < vec.size(); ++i) {
        vec[i] = static_cast<value_type>(0);
    }

    return vec;
}


/*
 * megamol::core::thecam::math::operator *=
 */
template <class V, size_t D, class T>
megamol::core::thecam::math::vector<V, D, T>& megamol::core::thecam::math::operator*=(
    vector<V, D, T>& lhs, const typename vector<V, D, T>::value_type rhs) {
    typedef typename T::size_type size_type;
    typedef typename T::value_type value_type;

    for (size_type i = 0; i < lhs.size(); ++i) {
        lhs[i] *= rhs;
    }

    return lhs;
}


/*
 * megamol::core::thecam::math::operator *=
 */
template <class V, size_t D, class T>
megamol::core::thecam::math::vector<V, D, T>& megamol::core::thecam::math::operator/=(
    vector<V, D, T>& lhs, const typename vector<V, D, T>::value_type rhs) {
    typedef typename T::size_type size_type;
    typedef typename T::value_type value_type;

    THE_ASSERT(rhs != static_cast<value_type>(0));
    for (size_type i = 0; i < lhs.size(); ++i) {
        lhs[i] /= rhs;
    }

    return lhs;
}


/*
 * megamol::core::thecam::math::operator +=
 */
template <class V1, class T1, class V2, class T2, size_t D>
megamol::core::thecam::math::vector<V1, D, T1> megamol::core::thecam::math::operator+=(
    vector<V1, D, T1>& lhs, const vector<V2, D, T2>& rhs) {
    typedef typename T1::size_type size_type;
    typedef typename T1::value_type value_type;

    for (size_type i = 0; i < lhs.size(); ++i) {
        lhs[i] += static_cast<value_type>(rhs[i]);
    }

    return lhs;
}


/*
 * megamol::core::thecam::math::operator -=
 */
template <class V1, class T1, class V2, class T2, size_t D>
megamol::core::thecam::math::vector<V1, D, T1> megamol::core::thecam::math::operator-=(
    vector<V1, D, T1>& lhs, const vector<V2, D, T2>& rhs) {
    typedef typename T1::size_type size_type;
    typedef typename T1::value_type value_type;

    for (size_type i = 0; i < lhs.size(); ++i) {
        lhs[i] -= static_cast<value_type>(rhs[i]);
    }

    return lhs;
}
