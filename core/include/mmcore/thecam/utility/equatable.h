/*
 * include\the\equatable.h
 *
 * Copyright (C) 2012 TheLib Team (http://www.thelib.org/license)
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

#ifndef THE_EQUATABLE_H_INCLUDED
#define THE_EQUATABLE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "mmcore/thecam/utility/config.h"


namespace megamol {
namespace core {
namespace thecam {
namespace utility {

/**
 * This abstract class implements a consistent behaviour of
 * operator == and operator != for classes that can provide a method
 * bool equals(const T& rhs) const.
 *
 * Subclasses (which must provide their type via the T parameter) only
 * must implement the equals() method with the above mentioned signature.
 *
 * This class uses curiously recurring template pattern (CRTP) to avoid the
 * necessity for a virtual table.
 *
 * @tparam T The type of the direct specialisation.
 */
template <class T> class equatable {

public:
    /** Dtor. */
    inline ~equatable(void) {}

    /**
     * Test for equality.
     *
     * @param rhs The right hand side operand.
     *
     * @return true if this object and 'rhs' are equal, false otherwise.
     */
    inline bool operator==(const T& rhs) const { return static_cast<const T*>(this)->equals(rhs); }

    /**
     * Test for inequality.
     *
     * @param rhs The right hand side operand.
     *
     * @return true if this object and 'rhs' are not equal, false otherwise.
     */
    inline bool operator!=(const T& rhs) const { return !static_cast<const T*>(this)->equals(rhs); }

protected:
    /**
     * Initialises a new instance.
     */
    inline equatable(void) {}

    /**
     * Create a clone of 'rhs'.
     *
     * @param rhs The object to be cloned.
     */
    inline equatable(const equatable& rhs) { *this = rhs; }

    /**
     * Assign values of 'rhs' to this object.
     *
     * @param rhs The right hand side operand.
     *
     * @return *this.
     */
    inline equatable& operator=(const equatable& rhs) { return *this; }
};

} /* end namespace utility*/
} /* end namespace thecam */
} /* end namespace core */
} /* end namespace megamol */

#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* THE_EQUATABLE_H_INCLUDED */
