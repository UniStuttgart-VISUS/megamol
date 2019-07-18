/*
 * thecam/processed_synchronisable_property.h
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

#ifndef THE_GRAPHICS_CAMERA_PROCESSED_SYNCHRONISABLE_PROPERTY_H_INCLUDED
#define THE_GRAPHICS_CAMERA_PROCESSED_SYNCHRONISABLE_PROPERTY_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "mmcore/thecam/utility/config.h"

#include <functional>
#include <memory>

#include "mmcore/thecam/utility/assert.h"
#include "mmcore/thecam/utility/types.h"

#include "mmcore/thecam/property_base.h"


namespace megamol {
namespace core {
namespace thecam {

/**
 * An implementation of a camera property which uses a shared value that can
 * be post-processed before being used.
 *
 * The implementation is the same as for synchronisable_property, but the
 * function call operator will invoke a user-defined callback before
 * returning any of the values. This implementation is by far the most
 * expensive wrt runtime. Its intended use is a camera rig which shares most
 * of the parameters, but requires a constant transformation on some of
 * them. An example would be a camera rig for dome rendering which shares
 * all of the camera paramters but needs to rotate the view direction.
 *
 * @tparam T The type that is used to pass the property around. The type
 *           that is stored is the decayed version of this type.
 */
template <class T> class processed_synchronisable_property : public detail::property_base<T> {

    /** The type used to store the actual value. */
    typedef typename detail::property_base<T>::value_type value_type;

    /** The type of the property that is used in the parameter list. */
    typedef typename detail::property_base<T>::parameter_type parameter_type;

    /** The post-processing functor which can be registered. */
    typedef std::function<value_type(const parameter_type)> processor_type;

    /**
     * Initialises a new, indepentent property.
     */
    inline processed_synchronisable_property(void) : value(std::make_shared<value_type>()) {}

    /**
     * Initialises a new instance that retrieves its value from 'rhs'.
     *
     * @param rhs           Another property which should be shared with
     *                      this one.
     * @param postProcessor The post-processing callback which will be
     *                      invoked when retrieving a value. It is safe to
     *                      pass an invalid target.
     */
    inline processed_synchronisable_property(
        const processed_synchronisable_property& rhs, const processor_type& postProcessor = processor_type())
        : postProcessorsor(postProcessor), value(rhs.value) {}

    /**
     * Initialises a new instance without providing storage for the
     * property's value.
     *
     * This constructor is only intended for internal use in the camera as
     * it requires the property being synchronised to another one before
     * making any access to the property's value.
     */
    inline processed_synchronisable_property(const megamol::core::thecam::utility::do_not_initialise_t)
        : value(nullptr) {}

    /**
     * Installs a new post-processing callback.
     *
     * @param postProcessor The post-processing callback which will be
     *                      invoked when retrieving a value. It is safe to
     *                      pass an invalid target.
     */
    inline void post_process(const processor_type& postProcessor) { this->postProcessor = postProcessor; }

    /**
     * Synchronises the property with another one.
     *
     * @param prop The property to synchronise with.
     */
    inline void synchronise(synchronisable_property<T>& prop) { // TODO is this correct?
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
        if (this->postProcessor) {
            return this->postProcessor(*this->value);
        } else {
            return *this->value;
        }
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
    /** The post-processing function. */
    processor_type postProcessor;

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
#endif /* THE_GRAPHICS_CAMERA_PROCESSED_SYNCHRONISABLE_PROPERTY_H_INCLUDED */
