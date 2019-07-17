/*
 * the/math/rectangle.inl
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
 * megamol::core::thecam::math::rectangle<V, T>::from_bottom_left
 */
template <class V, class T>
template <class P, class S>
megamol::core::thecam::math::rectangle<V, T> megamol::core::thecam::math::rectangle<V, T>::from_bottom_left(
    const P& point, const S& size) {
    return rectangle::from_bottom_left(static_cast<value_type>(point.x()), static_cast<value_type>(point.y()),
        static_cast<value_type>(size.width()), static_cast<value_type>(size.height()));
}


/*
 * megamol::core::thecam::math::rectangle<V, T>::from_top_left
 */
template <class V, class T>
template <class P, class S>
megamol::core::thecam::math::rectangle<V, T> megamol::core::thecam::math::rectangle<V, T>::from_top_left(
    const P& point, const S& size) {
    return rectangle::from_top_left(static_cast<value_type>(point.x()), static_cast<value_type>(point.y()),
        static_cast<value_type>(size.width()), static_cast<value_type>(size.height()));
}


/*
 * megamol::core::thecam::math::rectangle<V, T>::rectangle
 */
template <class V, class T> megamol::core::thecam::math::rectangle<V, T>::rectangle(void) {
    const auto ZERO = static_cast<value_type>(0);
    traits_type::assign(this->data, ZERO, ZERO, ZERO, ZERO);
}


/*
 * megamol::core::thecam::math::rectangle<V, T>::rectangle
 */
template <class V, class T>
megamol::core::thecam::math::rectangle<V, T>::rectangle(std::initializer_list<value_type> il) {
    size_type i = 0;
    auto it = il.begin();

    for (; (i < this->size_()) && (it != il.end()); ++i, ++it) {
        traits_type::at(this->data, i) = *it;
    }
    for (; i < this->size_(); ++i) {
        traits_type::at(this->data, i) = static_cast<value_type>(0);
    }
}


/*
 * megamol::core::thecam::math::rectangle<V, T>::empty
 */
template <class V, class T> bool megamol::core::thecam::math::rectangle<V, T>::empty(void) const {
    return ((this->left() == static_cast<value_type>(0)) && (this->top() == static_cast<value_type>(0)) &&
            (this->right() == static_cast<value_type>(0)) && (this->bottom() == static_cast<value_type>(0)));
}


/*
 * megamol::core::thecam::math::rectangle<V, T>::empty
 */
template <class V, class T> bool megamol::core::thecam::math::rectangle<V, T>::empty(const value_type epsilon) const {
    return ((std::abs(this->left()) < epsilon) && (std::abs(this->top()) < epsilon) &&
            (std::abs(this->right()) < epsilon) && (std::abs(this->bottom()) < epsilon));
}


/*
 * megamol::core::thecam::math::rectangle<V, T>::equals
 */
template <class V, class T>
template <class Vp, class Tp>
bool megamol::core::thecam::math::rectangle<V, T>::equals(
    const rectangle<Vp, Tp>& rhs, const value_type epsilon) const {
    return (is_equal(this->left(), static_cast<value_type>(rhs.left()), epsilon) &&
            is_equal(this->top(), static_cast<value_type>(rhs.top()), epsilon) &&
            is_equal(this->right(), static_cast<value_type>(rhs.right()), epsilon) &&
            is_equal(this->bottom(), static_cast<value_type>(rhs.bottom()), epsilon));
}


/*
 * megamol::core::thecam::math::rectangle<V, T>::operator =
 */
template <class V, class T>
megamol::core::thecam::math::rectangle<V, T>& megamol::core::thecam::math::rectangle<V, T>::operator=(
    const rectangle& rhs) {
    if (this != std::addressof(rhs)) {
        traits_type::copy(this->data, rhs.data);
    }
    return *this;
}


/*
 * megamol::core::thecam::math::rectangle<V, T>::operator =
 */
template <class V, class T>
template <class Vp, class Tp>
megamol::core::thecam::math::rectangle<V, T>& megamol::core::thecam::math::rectangle<V, T>::operator=(
    const rectangle<Vp, Tp>& rhs) {
    THE_ASSERT(this != std::addressof(rhs));
    traits_type::assign(this->data, static_cast<value_type>(rhs.left()), static_cast<value_type>(rhs.top()),
        static_cast<value_type>(rhs.right()), static_cast<value_type>(rhs.bottom()));
    return *this;
}
