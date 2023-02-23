/**
 * MegaMol
 * Copyright (c) 2006, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

namespace megamol::core::factories {

/**
 * Abstract base class for all object descriptions.
 *
 * An object is described using a unique name. This name is compared case insensitive!
 */
class ObjectDescription {
public:
    /**
     * Ctor.
     */
    ObjectDescription() = default;

    /**
     * Dtor.
     */
    virtual ~ObjectDescription() = default;

    /**
     * Answer the class name of the objects of this description.
     *
     * @return The class name of the objects of this description.
     */
    virtual const char* ClassName() const = 0;

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    virtual const char* Description() const = 0;
};

} // namespace megamol::core::factories
