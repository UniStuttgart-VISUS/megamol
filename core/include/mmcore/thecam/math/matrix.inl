/*
 * thecam\math\matrix.inl
 *
 * Copyright (C) 2016 TheLib Team (http://www.thelib.org/license)
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
 * megamol::core::thecam::math::matrix<V, R, C, L, T>::matrix
 */
template <class V, size_t R, size_t C, megamol::core::thecam::math::matrix_layout L, class T>
megamol::core::thecam::math::matrix<V, R, C, L, T>::matrix(std::initializer_list<value_type> il) {
    auto it = il.begin();
    for (size_type r = 0; r < this->rows(); ++r) {
        for (size_type c = 0; c < this->columns(); ++c) {
            if (it != il.end()) {
                traits_type::at(this->data, r, c) = *it++;
            } else {
                traits_type::at(this->data, r, c) = static_cast<value_type>(0);
            }
        }
    }
}


/*
 * megamol::core::thecam::math::matrix<V, R, C, L, T>::empty
 */
template <class V, size_t R, size_t C, megamol::core::thecam::math::matrix_layout L, class T>
bool megamol::core::thecam::math::matrix<V, R, C, L, T>::empty(const value_type epsilon) const {
    for (size_type r = 0; r < this->rows(); ++r) {
        for (size_type c = 0; c < this->columns(); ++c) {
            if (!is_equal(traits_type::at(this->data, r, c), static_cast<value_type>(0), epsilon)) {
                return false;
            }
        }
    }
    return true;
}


/*
 * megamol::core::thecam::math::matrix<V, R, C, L, T>::equals
 */
template <class V, size_t R, size_t C, megamol::core::thecam::math::matrix_layout L, class T>
template <class Vp, size_t Rp, size_t Cp, megamol::core::thecam::math::matrix_layout Lp, class Tp>
bool megamol::core::thecam::math::matrix<V, R, C, L, T>::equals(
    const matrix<Vp, Rp, Cp, Lp, Tp>& rhs, const value_type epsilon) const {
    if ((R == Rp) && (C == Cp)) {
        for (size_type r = 0; r < this->rows(); ++r) {
            for (size_type c = 0; c < this->columns(); ++c) {
                if (!is_equal(traits_type::at(this->data, r, c), rhs(r, c), epsilon)) {
                    return false;
                }
            }
        }
        return true;

    } else {
        // Trivial reject: Matrices have different dimensions and hence cannot
        // be equal.
        return false;
    }
}


/*
 * megamol::core::thecam::math::matrix<V, R, C, L, T>::identity
 */
template <class V, size_t R, size_t C, megamol::core::thecam::math::matrix_layout L, class T>
bool megamol::core::thecam::math::matrix<V, R, C, L, T>::identity(const value_type epsilon) const {
    if (R == C) {
        for (size_type r = 0; r < this->rows(); ++r) {
            for (size_type c = 0; c < this->columns(); ++c) {
                auto e = (r == c) ? static_cast<value_type>(1) : static_cast<value_type>(0);
                if (!is_equal(traits_type::at(this->data, r, c), e, epsilon)) {
                    return false;
                }
            }
        }
        return true;

    } else {
        // Trivial reject: If the dimensions are different, this cannot be an
        // identity matrix.
        return false;
    }
}


/*
 * megamol::core::thecam::math::matrix<V, R, C, L, T>::operator =
 */
template <class V, size_t R, size_t C, megamol::core::thecam::math::matrix_layout L, class T>
megamol::core::thecam::math::matrix<V, R, C, L, T>& megamol::core::thecam::math::matrix<V, R, C, L, T>::operator=(
    const matrix& rhs) {
    if (this != std::addressof(rhs)) {
        traits_type::copy(this->data, rhs.data);
    }

    return *this;
}


/*
 * megamol::core::thecam::math::matrix<V, R, C, L, T>::operator =
 */
template <class V, size_t R, size_t C, megamol::core::thecam::math::matrix_layout L, class T>
template <class Vp, megamol::core::thecam::math::matrix_layout Lp, class Tp>
megamol::core::thecam::math::matrix<V, R, C, L, T>& megamol::core::thecam::math::matrix<V, R, C, L, T>::operator=(
    const matrix<Vp, R, C, Lp, Tp>& rhs) {
    THE_ASSERT(this != std::addressof(rhs));
    for (size_type r = 0; r < R; ++r) {
        for (size_type c = 0; c < C; ++c) {
            (*this)(r, c) = rhs(r, c);
        }
    }
    return *this;
}


/*
 * megamol::core::thecam::math::det
 */
template <class V, megamol::core::thecam::math::matrix_layout L, class T>
typename T::value_type megamol::core::thecam::math::det(const matrix<V, 4, 4, L, T>& matrix) {
    megamol::core::thecam::math::matrix<V, 3, 3, L, T> m1 = {matrix(1, 0), matrix(2, 0), matrix(3, 0), matrix(1, 1),
        matrix(2, 1), matrix(3, 1), matrix(1, 3), matrix(2, 3), matrix(3, 3)};
    megamol::core::thecam::math::matrix<V, 3, 3, L, T> m2 = {matrix(0, 0), matrix(2, 0), matrix(3, 0), matrix(0, 1),
        matrix(2, 1), matrix(3, 1), matrix(0, 3), matrix(2, 3), matrix(3, 3)};
    megamol::core::thecam::math::matrix<V, 3, 3, L, T> m3 = {matrix(0, 0), matrix(1, 0), matrix(3, 0), matrix(0, 1),
        matrix(1, 1), matrix(3, 1), matrix(0, 3), matrix(1, 3), matrix(3, 3)};
    megamol::core::thecam::math::matrix<V, 3, 3, L, T> m4 = {matrix(0, 0), matrix(1, 0), matrix(2, 0), matrix(0, 1),
        matrix(1, 1), matrix(2, 1), matrix(0, 3), matrix(1, 3), matrix(2, 3)};

    return (matrix(0, 2) * det(m1) - matrix(1, 2) * det(m2) + matrix(2, 2) * det(m3) - matrix(3, 2) * det(m4));
}


/*
 * megamol::core::thecam::math::invert
 */
template <class V, size_t D, megamol::core::thecam::math::matrix_layout L, class T>
bool megamol::core::thecam::math::invert(matrix<V, D, D, L, T>& matrix) {
#define A(r, c) a[(r)*2 * D + (c)]
    typedef megamol::core::thecam::math::matrix<V, D, D, L, T> matrix_type;
    typedef typename matrix_type::size_type size_type;
    typedef typename matrix_type::value_type value_type;
    double a[2 * D * D]; // Input matrix for algorithm.
    double f;            // Multiplication factor.
    double maxVal;       // Row pivotising.
    size_type pRow;      // Pivot row.
    size_type s;         // Current eliminination step.

    /* Create double precision matrix and add identity at the right. */
    for (size_type r = 0; r < D; r++) {
        for (size_type c = 0; c < D; c++) {
            A(r, c) = static_cast<double>(matrix(r, c));
        }

        for (size_type c = 0; c < D; c++) {
            A(r, c + D) = (r == c) ? 1.0 : 0.0;
        }
    }

    /* Gauß elimination. */
    s = 0;
    do {
        // Pivotising avoids unnecessary cancelling if a zero is in the
        // diagonal and increases the precision.
        maxVal = ::fabs(A(s, s));
        pRow = s;
        for (size_type r = s + 1; r < D; r++) {
            if (::fabs(A(r, s)) > maxVal) {
                maxVal = ::fabs(A(r, s));
                pRow = r;
            }
        }

        if (maxVal < megamol::core::thecam::math::epsilon<double>::value) {
            return false; // delete is not possible
        }

        if (pRow != s) {
            // if necessary, exchange the row
            double h;

            for (size_type c = s; c < 2 * D; c++) {
                h = A(s, c);
                A(s, c) = A(pRow, c);
                A(pRow, c) = h;
            }
        }

        // eliminations row is divided by pivot-coefficient f = a[s][s]
        f = A(s, s);
        for (size_type c = s; c < 2 * D; c++) {
            A(s, c) /= f;
        }

        for (size_type r = 0; r < D; r++) {
            if (r != s) {
                f = -A(r, s);
                for (size_type c = s; c < 2 * D; c++) {
                    A(r, c) += f * A(s, c);
                }
            }
        }

        s++;
    } while (s < D);

    /* Copy identity on the right which is now inverse. */
    for (size_type r = 0; r < D; r++) {
        for (size_type c = 0; c < D; c++) {
            matrix(r, c) = static_cast<value_type>(A(r, D + c));
        }
    }

    return true;
#undef A
}


/*
 * megamol::core::thecam::math::set_empty
 */
template <class V, size_t R, size_t C, megamol::core::thecam::math::matrix_layout L, class T>
megamol::core::thecam::math::matrix<V, R, C, L, T>& megamol::core::thecam::math::set_empty(
    megamol::core::thecam::math::matrix<V, R, C, L, T>& matrix) {
    typedef typename T::size_type size_type;
    typedef typename T::value_type value_type;

    for (size_type r = 0; r < R; ++r) {
        for (size_type c = 0; c < C; ++c) {
            matrix(r, c) = static_cast<value_type>(0);
        }
    }

    return matrix;
}


/*
 * megamol::core::thecam::math::set_identity
 */
template <class V, size_t D, megamol::core::thecam::math::matrix_layout L, class T>
megamol::core::thecam::math::matrix<V, D, D, L, T>& megamol::core::thecam::math::set_identity(
    megamol::core::thecam::math::matrix<V, D, D, L, T>& matrix) {
    typedef typename T::size_type size_type;
    typedef typename T::value_type value_type;

    for (size_type r = 0; r < D; ++r) {
        for (size_type c = 0; c < D; ++c) {
            matrix(r, c) = static_cast<value_type>((r == c) ? 1 : 0);
        }
    }

    return matrix;
}


/*
 * megamol::core::thecam::math::trace
 */
template <class V, size_t D, megamol::core::thecam::math::matrix_layout L, class T>
typename T::value_type megamol::core::thecam::math::trace(const matrix<V, D, D, L, T>& matrix) {
    typedef typename T::size_type size_type;
    typedef typename T::value_type value_type;

    value_type retval = static_cast<value_type>(0);

    for (size_type d = 0; d < D; ++d) {
        retval += matrix(d, d);
    }

    return retval;
}


/*
 * megamol::core::thecam::math::transpose
 */
template <class V, size_t R, size_t C, megamol::core::thecam::math::matrix_layout L, class T>
megamol::core::thecam::math::matrix<V, C, R, L, T> megamol::core::thecam::math::transpose(
    const matrix<V, R, C, L, T>& matrix) {
    typedef typename T::size_type size_type;
    typedef typename T::value_type value_type;

    megamol::core::thecam::math::matrix<V, C, R, L, T> retval(megamol::core::thecam::utility::do_not_initialise);

    for (size_type r = 0; r < R; ++r) {
        for (size_type c = 0; c < C; ++c) {
            retval(c, r) = matrix(r, c);
        }
    }

    return std::move(retval);
}


/*
 * megamol::core::thecam::math::operator *=
 */
template <class V, size_t R, size_t C, megamol::core::thecam::math::matrix_layout L, class T>
megamol::core::thecam::math::matrix<V, R, C, L, T>& megamol::core::thecam::math::operator*=(
    matrix<V, R, C, L, T>& lhs, const typename matrix<V, R, C, L, T>::value_type rhs) {
    typedef typename T::size_type size_type;
    typedef typename T::value_type value_type;

    for (size_type r = 0; r < R; ++r) {
        for (size_type c = 0; c < C; ++c) {
            lhs(r, c) *= rhs;
        }
    }

    return lhs;
}


/*
 * operator *=
 */
template <class VL, class VR, size_t RL, size_t CL, size_t CR, megamol::core::thecam::math::matrix_layout LL,
    megamol::core::thecam::math::matrix_layout LR, class TL, class TR>
megamol::core::thecam::math::matrix<VL, RL, CR, LL, megamol::core::thecam::math::matrix_traits<VL, RL, CR, LL>>
    megamol::core::thecam::math::operator*(
        const matrix<VL, RL, CL, LL, TL>& lhs, const matrix<VR, CL, CR, LR, TR>& rhs) {
    typedef typename TL::size_type size_type;
    typedef typename TL::value_type value_type;

    megamol::core::thecam::math::matrix<VL, RL, CR, LL> retval;
    THE_ASSERT(retval.size() == RL * CR);
    THE_ASSERT(retval.empty());

    for (size_type r = 0; r < RL; r++) {
        for (size_type c = 0; c < CR; c++) {
            for (size_type i = 0; i < CL; i++) {
                retval(r, c) += lhs(r, i) * rhs(i, c);
            }
        }
    }

    return std::move(retval);
}


/*
 * megamol::core::thecam::math::operator *
 */
template <class VM, class VV, size_t R, size_t C, megamol::core::thecam::math::matrix_layout L, class TM, class TV>
megamol::core::thecam::math::vector<typename TM::value_type, R> megamol::core::thecam::math::operator*(
    const matrix<VM, R, C, L, TM>& lhs, const vector<VV, C, TV>& rhs) {
    typedef typename TM::size_type size_type;
    typedef typename TM::value_type value_type;

    megamol::core::thecam::math::vector<value_type, R> retval;
    THE_ASSERT(retval.empty());

    for (size_type r = 0; r < R; ++r) {
        for (size_type c = 0; c < C; ++c) {
            retval[r] += static_cast<value_type>(lhs(r, c) * rhs[c]);
        }
    }

    return std::move(retval);
}


/*
 * megamol::core::thecam::math::operator *
 */
template <class VM, class VV, size_t R, size_t C, megamol::core::thecam::math::matrix_layout L, class TM, class TV>
megamol::core::thecam::math::vector<typename TM::value_type, C> megamol::core::thecam::math::operator*(
    const vector<VV, R, TV>& rhs, const matrix<VM, R, C, L, TM>& lhs) {
    typedef typename TM::size_type size_type;
    typedef typename TM::value_type value_type;

    megamol::core::thecam::math::vector<value_type, C> retval;
    THE_ASSERT(retval.empty());

    for (size_type r = 0; r < R; ++r) {
        for (size_type c = 0; c < C; ++c) {
            retval[c] += static_cast<value_type>(lhs(r, c) * rhs[r]);
        }
    }

    return std::move(retval);
}
