/*
 * thecam/property.h
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

#ifndef THE_GRAPHICS_CAMERA_PROPERTY_H_INCLUDED
#define THE_GRAPHICS_CAMERA_PROPERTY_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "mmcore/thecam/utility/config.h"

#include "mmcore/thecam/property_base.h"


namespace megamol {
namespace core {
namespace thecam {

/**
 * Provides the simplest form of a camera property.
 *
 * This kind of properties should be used for standalone cameras that do not
 * need to be synchronised in a camera_rig, because they introduce the least
 * overhead. For more complex scenarios like stereo rendering and tiled
 * displays, use synchronisable_property to synchronise and selectively
 * override properties.
 *
 * @tparam T The type that is used to pass the property around. The type
 *           that is stored is the decayed version of this type.
 */
template <class T> class property : public detail::property_base<T> {

public:
    /** The type used to store the actual value. */
    typedef typename detail::property_base<T>::value_type value_type;

    /** The type of the property that is used in the parameter list. */
    typedef typename detail::property_base<T>::parameter_type parameter_type;

    /**
     * Gets the current value of the property.
     *
     * @return The current value of the property.
     */
    inline parameter_type operator()(void) const { return this->value; }

    // inline parameter_type& operator ()(void) {
    //    return this->value;
    //}

    /**
     * Sets a new property value.
     *
     * Rationale: We use the function call operator rather than the
     * assignment operator for setting a new value, because derived
     * properties, which are computed in the camera, can only be implemented
     * using function call semantics. Therefore, using the function call
     * operator provides a more consistent user experience.
     *
     * @param value The new property value.
     */
    inline void operator()(parameter_type value) { this->value = value; }

private:
    /** The value of the property. */
    value_type value;
};

} /* end namespace thecam */
} /* end namespace core */
} /* end namespace megamol */


#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* THE_GRAPHICS_CAMERA_PROPERTY_H_INCLUDED */
