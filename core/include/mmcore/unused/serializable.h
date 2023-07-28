/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <string>

namespace megamol::core {

/**
 * Interface for serializable classes.
 */
class serializable {
public:
    /**
     * Serializes the state of this object into a string description.
     *
     * @return the description of this object.
     */
    virtual std::string Serialize() const = 0;

    /**
     * Deserializes a string description into the state of this object.
     *
     * @param descr the string description of an object of this type
     */
    virtual void Deserialize(std::string const& descr) = 0;

    /** dtor */
    virtual ~serializable() = default;
}; // end class serializable

} // namespace megamol::core
