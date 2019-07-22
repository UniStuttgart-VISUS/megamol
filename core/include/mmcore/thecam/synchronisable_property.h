/*
 * thecam/synchronisable_property.h
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

#ifndef THE_GRAPHICS_CAMERA_SYNCHRONISABLE_PROPERTY_H_INCLUDED
#define THE_GRAPHICS_CAMERA_SYNCHRONISABLE_PROPERTY_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "mmcore/thecam/utility/config.h"

#include <memory>

#include "mmcore/thecam/utility/assert.h"
#include "mmcore/thecam/utility/types.h"

#include "mmcore/thecam/property_base.h"


namespace megamol {
namespace core {
namespace thecam {

/**
 * An implementation of a camera property which enables sharing settings
 * between different camera instances.
 *
 * The implementation used a heap-allocated, reference-counted memory block
 * to store the actual property value. This block can be shared between
 * multiple instances. If you only need a single camera, it is recommended
 * to use property instead to avoid the performance overhead for
 * synchronisable properties.
 *
 * @tparam T The type that is used to pass the property around. The type
 *           that is stored is the decayed version of this type.
 */
template <class T> class synchronisable_property : public detail::property_base<T> {

    /** The type used to store the actual value. */
    typedef typename detail::property_base<T>::value_type value_type;

    /** The type of the property that is used in the parameter list. */
    typedef typename detail::property_base<T>::parameter_type parameter_type;

    /**
     * Initialises a new, indepentent property.
     */
    inline synchronisable_property(void) : value(std::make_shared<value_type>()) {}

    /**
     * Initialises a new instance that retrieves its value from 'rhs'.
     *
     * @param rhs Another property which should be shared with this one.
     */
    inline synchronisable_property(const synchronisable_property& rhs) : value(rhs.value) {}

    /**
     * Initialises a new instance without providing storage for the
     * property's value.
     *
     * This constructor is only intended for internal use in the camera as
     * it requires the property being synchronised to another one before
     * making any access to the property's value.
     */
    inline synchronisable_property(const megamol::core::thecam::utility::do_not_initialise_t) : value(nullptr) {}

    /**
     * Synchronises the property with another one.
     *
     * @param prop The property to synchronise with.
     */
    inline void synchronise(synchronisable_property& prop) {
        THE_ASSERT(prop.value);
        this->value = prop.value;
    }

    /**
     * Unlinks the property from any other one.
     */
    inline void unsynchronise(void) {
        auto tmp = this->value;
        this->value = std::make_shared<value_type>(*tmp);
    }

    /**
     * Gets the current value of the property.
     *
     * @return The current value of the property.
     */
    inline const parameter_type operator()(void) const {
        THE_ASSERT(this->value != nullptr);
        return *this->value;
    }

    /**
     * Sets a new property value.
     *
     * @param value The new property value.
     */
    inline void operator()(const parameter_type value) {
        THE_ASSERT(this->value != nullptr);
        *this->value = value;
    }

private:
    /**
     * The value of the property, which might be shared with other instances
     * of synchronisable_property.
     */
    std::shared_ptr<value_type> value;
};

} /* end namespace thecam */
} /* end namespace core */
} /* end namespace megamol */


#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* THE_GRAPHICS_CAMERA_SYNCHRONISABLE_PROPERTY_H_INCLUDED */
