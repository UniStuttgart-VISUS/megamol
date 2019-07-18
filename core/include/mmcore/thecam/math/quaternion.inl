/*
 * thecam\math\quaternion_base.inl
 *
 * Copyright (C) 2016 - 2017 TheLib Team (http://www.thelib.org/license)
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
 * megamol::core::thecam::math::quaternion<V, T>::quaternion
 */
template <class V, class T>
megamol::core::thecam::math::quaternion<V, T>::quaternion(std::initializer_list<value_type> il) {
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
 * megamol::core::thecam::math::detail::quaternion<V, T>::operator =
 */
template <class V, class T>
megamol::core::thecam::math::quaternion<V, T>& megamol::core::thecam::math::quaternion<V, T>::operator=(
    const quaternion& rhs) {
    if (this != std::addressof(rhs)) {
        traits_type::copy(this->data, rhs.data);
    }

    return *this;
}


/*
 * megamol::core::thecam::math::invert
 */
template <class V, class T>
megamol::core::thecam::math::quaternion<V, T> megamol::core::thecam::math::invert(const quaternion<V, T>& quat) {
    typedef typename T::value_type value_type;

    auto l = square_norm(quat);

    if (l != static_cast<value_type>(0)) {
        l = sqrt(l);
        quaternion<V, T> retval(quat.x() /= -l, quat.y() /= -l, quat.z() /= -l, quat.w() / l);
        return std::move(retval);

    } else {
        quaternion<V, T> retval(static_cast<value_type>(0), static_cast<value_type>(0), static_cast<value_type>(0),
            static_cast<value_type>(1));
        THE_ASSERT(retval.identity());
        return retval;
    }
}


/*
 * megamol::core::thecam::math::normalise
 */
template <class V, class T>
megamol::core::thecam::math::quaternion<V, T> megamol::core::thecam::math::normalise(const quaternion<V, T>& quat) {
    typedef typename T::value_type value_type;

    auto l = square_norm(quat);

    if (l != static_cast<value_type>(0)) {
        l = sqrt(l);
        quaternion<V, T> retval(quat.x() / l, quat.y() / l, quat.z() / l, quat.w() / l);
        return std::move(retval);

    } else {
        quaternion<V, T> retval(static_cast<value_type>(0), static_cast<value_type>(0), static_cast<value_type>(0),
            static_cast<value_type>(1));
        THE_ASSERT(retval.identity());
        return retval;
    }
}


/*
 * megamol::core::thecam::math::rotate
 */
template <class V, class TQ, class TV>
inline megamol::core::thecam::math::vector<V, 3, TV> megamol::core::thecam::math::rotate(
    const vector<V, 3, TV>& vec, const quaternion<V, TQ>& quat) {
    typedef typename TQ::value_type value_type;
    const auto zero = static_cast<value_type>(0);
    quaternion<V, TQ> a(vec.x(), vec.y(), vec.z(), zero);
    quaternion<V, TQ> c = conjugate(quat);
    auto retval = quat * a * c;
    return vector<V, 3, TV>(retval.x(), retval.y(), retval.z());
}


/*
 * megamol::core::thecam::math::set_from_angle_axis
 */
template <class Q, class V, class A>
megamol::core::thecam::math::quaternion<Q>& megamol::core::thecam::math::set_from_angle_axis(
    quaternion<Q>& quat, const A angle, const vector<V, 3>& axis) {
    typedef typename quaternion<Q>::value_type value_type;
    value_type len = length(axis);

    if (!is_equal(len, static_cast<value_type>(0))) {
        auto a = 0.5 * static_cast<double>(angle);
        auto s = sin(a);
        quat.assign(static_cast<value_type>(axis[0] * s), static_cast<value_type>(axis[1] * s),
            static_cast<value_type>(axis[2] * s), static_cast<value_type>(::cos(a)));

    } else {
        set_identity(quat);
    }

    return quat;
}


/*
 * megamol::core::thecam::math::set_from_vectors
 */
template <class Q, class V>
megamol::core::thecam::math::quaternion<Q>& megamol::core::thecam::math::set_from_vectors(
    quaternion<Q>& quat, const vector<V, 3>& u, const vector<V, 3>& v) {
    const auto ONE = static_cast<V>(1);
    const auto TWO = static_cast<V>(2);
    auto m = sqrt(TWO + TWO * dot(u, v));
    auto w = (ONE / m) * cross(u, v);
    quat = quaternion<Q>(w.x(), w.y(), w.z(), m / TWO);
    return quat;
}


/*
 * megamol::core::thecam::math::set_from_vectors
 */
template <class Q, class V>
megamol::core::thecam::math::quaternion<Q>& megamol::core::thecam::math::set_from_vectors(
    quaternion<Q>& quat, const vector<V, 4>& u, const vector<V, 4>& v) {
    auto uu = vector<V, 3>(u.x(), u.y(), u.z());
    auto vv = vector<V, 3>(v.x(), v.y(), v.z());
    return set_from_vectors(quat, uu, vv);
}

#ifdef WITH_THE_GLM
/*
 * megamol::core::thecam::math::set_from_vectors
 */
inline megamol::core::thecam::math::quaternion<glm::quat>& megamol::core::thecam::math::set_from_vectors(
    quaternion<glm::quat>& quat, const vector<glm::vec3>& u, const vector<glm::vec3>& v) {
    auto m = sqrt(2.0f + 2.0f * glm::dot(static_cast<glm::vec3>(u), static_cast<glm::vec3>(v)));
    auto w = (1.0f / m) * glm::cross(static_cast<glm::vec3>(u), static_cast<glm::vec3>(v));
    quat = glm::quat(m / 2.0f, w.x, w.y, w.z); // TODO is this correct?
    return quat;
}
#endif /* WITH_THE_GLM */

#ifdef WITH_THE_GLM
/*
 * megamol::core::thecam::math::set_from_vectors
 */
inline megamol::core::thecam::math::quaternion<glm::quat>& megamol::core::thecam::math::set_from_vectors(
    quaternion<glm::quat>& quat, const vector<glm::vec4>& u, const vector<glm::vec4>& v) {
    auto uu = vector<glm::vec3>(u.x(), u.y(), u.z());
    auto vv = vector<glm::vec3>(v.x(), v.y(), v.z());
    return set_from_vectors(quat, uu, vv);
}
#endif /* WITH_THE_GLM */

/*
 * megamol::core::thecam::math::operator *
 */
template <class V, class T>
megamol::core::thecam::math::quaternion<V, T> megamol::core::thecam::math::operator*(
    const quaternion<V, T>& lhs, const quaternion<V, T>& rhs) {
    megamol::core::thecam::math::quaternion<V, T> retval(
        lhs.w() * rhs.x() + lhs.x() * rhs.w() + lhs.y() * rhs.z() - lhs.z() * rhs.y(),

        lhs.w() * rhs.y() + lhs.y() * rhs.w() + lhs.z() * rhs.x() - lhs.x() * rhs.z(),

        lhs.w() * rhs.z() + lhs.z() * rhs.w() + lhs.x() * rhs.y() - lhs.y() * rhs.x(),

        lhs.w() * rhs.w() - lhs.x() * rhs.x() - lhs.y() * rhs.y() - lhs.z() * rhs.z());
    return std::move(retval);
}
