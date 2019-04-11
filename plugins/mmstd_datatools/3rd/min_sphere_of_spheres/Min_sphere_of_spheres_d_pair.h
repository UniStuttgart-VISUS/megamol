// Copyright (c) 1997  ETH Zurich (Switzerland).
// All rights reserved.
//
// This file is part of CGAL (www.cgal.org); you may redistribute it under
// the terms of the Q Public License version 1.0.
// See the file LICENSE.QPL distributed with CGAL.
//
// Licensees holding a valid commercial license may use this file in
// accordance with the commercial license agreement provided with the software.
//
// This file is provided AS IS with NO WARRANTY OF ANY KIND, INCLUDING THE
// WARRANTY OF DESIGN, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
//
// $Source$
// $Revision$ $Date$
// $Name$
//
// Author(s)     : Kaspar Fischer


#ifndef CGAL_MINIBALL_PAIR
#define CGAL_MINIBALL_PAIR

namespace CGAL_MINIBALL_NAMESPACE {

namespace Min_sphere_of_spheres_d_impl {
const double Min_float = 1.0e-120;
const double Eps = 1.0e-16;
const double SqrOfEps = 1.0e-32;
const double Tol = 1.0 + Eps;
} // namespace Min_sphere_of_spheres_d_impl

template <typename FT> inline FT sqr(const FT& x) { return x * x; }

// We do not use std::inner_product() because in our case the number n
// of operands is known.  An optimizing compiler like GCC thus unrolls
// the following loop.  The same holds for copy_n().
template <int N, typename T, typename InputIterator1, typename InputIterator2, typename BinaryOperation1,
    typename BinaryOperation2>
inline T inner_product_n(
    InputIterator1 first1, InputIterator2 first2, T init, BinaryOperation1 op1, BinaryOperation2 op2) {
    for (int i = 0; i < N; ++i, ++first1, ++first2) init = op1(init, op2(*first1, *first2));
    return init;
}

template <int N, typename InputIterator, typename OutputIterator>
inline void copy_n(InputIterator first, OutputIterator result) {
    for (int i = 0; i < N; ++i, ++first, ++result) *result = *first;
}

inline bool is_approximate(const Tag_false is_exact) { return true; }

inline bool is_approximate(const Tag_true is_exact) { return false; }

template <typename FT> class Pair : public std::pair<FT, FT> {
private:
    typedef std::pair<FT, FT> Base;

public: // construction:
    Pair() : Base() {}

    Pair(const FT& a, const FT& b) : Base(a, b) {}

    Pair(int i) : Base(i, 0) {}

    Pair& operator=(const FT& x) {
        this->first = x;
        this->second = 0;
        return *this;
    }

public: // arithmetic and comparision:
    inline Pair operator+(const Pair& a) const { return Pair(this->first + a.first, this->second + a.second); }

    inline Pair operator-(const Pair& a) const { return Pair(this->first - a.first, this->second - a.second); }

    inline Pair operator-(const FT& a) const { return Pair(this->first - a, this->second); }

    inline Pair operator*(const FT& a) const { return Pair(this->first * a, this->second * a); }

    inline Pair operator/(const FT& a) const {
        CGAL_MINIBALL_ASSERT(a != FT(0));
        return Pair(this->first / a, this->second / a);
    }

    inline Pair& operator+=(const Pair& p) {
        this->first += p.first;
        this->second += p.second;
        return *this;
    }

    inline Pair& operator-=(const Pair& p) {
        this->first -= p.first;
        this->second -= p.second;
        return *this;
    }

    inline bool operator!=(const Pair& p) const { return this->first != p.first || this->second != p.second; }
};

template <typename FT> inline Pair<FT> operator+(const FT& a, const Pair<FT>& p) {
    return Pair<FT>(a + p.first, p.second);
}

template <typename FT> inline Pair<FT> operator-(const FT& a, const Pair<FT>& p) {
    return Pair<FT>(a - p.first, -p.second);
}

template <typename FT> inline bool is_neg(const FT& p, const FT&) { return p < 0; }

template <typename FT> inline bool is_neg(const Pair<FT> p, const FT& d) {
    const bool aneg = p.first < FT(0), bneg = p.second < FT(0);

    if (aneg && bneg) return true;
    if (!aneg && !bneg) return false;

    // So what remains are the cases (i) a<0,b>=0 and (ii) a>=0,b<0:
    //   (i)  We need to test b*sqrt(d)<-a with b,-a>=0.
    //   (ii) We need to test a<(-b)*sqrt(d) with a,-b>=0.
    // Hence:
    const FT x = sqr(p.second) * d, y = sqr(p.first);
    return aneg ? x < y : y < x;
}

template <typename FT> inline bool is_zero(const Pair<FT> p, const FT& d) {
    if (d != FT(0))
        // check whether the sides of a=-b*sqrt(d) (*)
        // have different signs:
        if ((p.first > FT(0)) ^ (p.second < FT(0))) return false;

    // Here we have either:
    //   (i)   d=0, or
    //   (ii)  a>0,b<0,d!=0, or
    //   (iii) a<=0,b>=0,d!=0
    // Hence both sides of (*) are either positive or negative.
    return sqr(p.first) == sqr(p.second) * d;
}

template <typename FT> inline bool is_neg_or_zero(const FT& p, const FT& d) { return p <= 0; }

template <typename FT> inline bool is_neg_or_zero(const Pair<FT> p, const FT& d) {
    return is_neg(p, d) || is_zero(p, d);
}

} // namespace CGAL_MINIBALL_NAMESPACE

#endif // CGAL_MINIBALL_PAIR
