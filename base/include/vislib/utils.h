/*
 * utils.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_UTILS_H_INCLUDED
#define VISLIB_UTILS_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */

namespace vislib {

    /**
     * Swaps the values of left and right.
     *
     * @param left  A reference to a variable to be swapped
     * @param right A reference to a variable to be swapped
     */
    template<class T> void Swap(T &left, T &right) {
        T tmp; 
        tmp = left; 
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

} /* end namespace vislib */

#endif /* VISLIB_UTILS_H_INCLUDED */
