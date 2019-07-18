/*
 * the/not_copyable.h
 *
 * Copyright (c) 2012, TheLib Team (http://www.thelib.org/license)
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

#ifndef THE_NOT_COPYABLE_H_INCLUDED
#define THE_NOT_COPYABLE_H_INCLUDED
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
 * Super class for all classes which do not support deep copies.
 *
 * Note that this class does not prevent move construction if the sub-class
 * intends to provide a move constructor. The move constructor of a derived
 * class D should look like
 * <code>
 * D::D(D&& rhs) : not_copyable(std::move(rhs)) { ... }
 * </code>
 */
class not_copyable {

public:
    /**
     * Dtor.
     */
    virtual ~not_copyable(void);

protected:
    /**
     * Initialises a new instance.
     */
    inline not_copyable(void) {}

    /**
     * Move 'rhs' to this object.
     *
     * Note: The destructor prevents the compiler from generating an
     * implicit move constructor, so we provide this explicit one.
     *
     * @param rhs The object to be moved.
     */
    inline not_copyable(not_copyable&& rhs) {}

private:
    /**
     * Forbidden copy ctor.
     *
     * @param rhs The object to be cloned.
     *
     * @throws megamol::core::thecam::not_supported_exception Unconditionally.
     */
    not_copyable(const not_copyable& rhs);

    /**
     * Forbidden assignment operator.
     *
     * @param rhs The right hand side operand.
     *
     * @return *this.
     *
     * @throws megamol::core::thecam::argument_exception If 'rhs' is not *this.
     */
    not_copyable& operator=(const not_copyable& rhs);
};

} /* end namespace utility */
} /* end namespace thecam */
} /* end namespace core */
} /* end namespace megamol */


#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* THE_NOT_COPYABLE_H_INCLUDED */
