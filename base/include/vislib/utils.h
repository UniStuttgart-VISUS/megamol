/*
 * utils.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_UTILS_H_INCLUDED
#define VISLIB_UTILS_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


namespace vislib {

    /**
     * Swaps the values of left and right.
     *
     * @param left  A reference to a variable to be swapped
     * @param right A reference to a variable to be swapped
     */
    template<class T> void Swap(T &left, T &right) {
        T tmp = left;
        left = right;
        right = tmp;
    }

    /**
     * Swaps the values of left and right. 
     * Uses the specified temporary variable.
     *
     * @param left  A reference to a variable to be swapped
     * @param right A reference to a variable to be swapped
     * @param tmp   A reference to a temporary variable.
     */
    template<class T> void Swap(T &left, T &right, T &tmp) {
        tmp = left; 
        left = right; 
        right = tmp;
    }

    /**
     * A comparator using the operator '-'. You can use this comparator for
     * sorting collections of basic types.
     *
     * @param lhs The left hand side operand
     * @param rhs The right hand side operand
     *
     * @return (lhs - rhs)
     *          = 0 if lhs == rhs
     *          < 0 if lhs < rhs
     *          > 0 if lhs > rhs
     */
    template<class T> int DiffComparator(const T& lhs, const T& rhs) {
        T diff = (lhs - rhs);
        return (diff < 0) ? -1 : ((diff > 0) ? 1 : 0);
    }

} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_UTILS_H_INCLUDED */
