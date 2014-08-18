/*
 * Serialiser.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/Serialiser.h"


/*
 * vislib::Serialiser::SERIALISER_SUPPORTS_NAMES 
 */
const UINT32 vislib::Serialiser::SERIALISER_SUPPORTS_NAMES = 0x00000001;


/*
 * vislib::Serialiser::SERIALISER_REQUIRES_NAMES
 */
const UINT32 vislib::Serialiser::SERIALISER_REQUIRES_NAMES = 0x00000002
    | vislib::Serialiser::SERIALISER_SUPPORTS_NAMES;


/*
 * vislib::Serialiser::SERIALISER_REQUIRES_ORDER
 */
const UINT32 vislib::Serialiser::SERIALISER_REQUIRES_ORDER = 0x00000004;


/*
 * vislib::Serialiser::~Serialiser
 */
vislib::Serialiser::~Serialiser(void) {
    // Nothing to do.
}


/*
 * vislib::Serialiser::Serialiser
 */
vislib::Serialiser::Serialiser(const UINT32 properties) 
        : properties(properties) {
    // Nothing to do.
}


/*
 * vislib::Serialiser::Serialiser
 */
vislib::Serialiser::Serialiser(const Serialiser& rhs) 
        : properties(rhs.properties) {
    // Nothing to do.
}


/*
 * vislib::Serialiser::operator =
 */
vislib::Serialiser& vislib::Serialiser::operator =(
        const Serialiser& rhs) {
    if (this != &rhs) {
        this->properties = rhs.properties;
    }
    return *this;
}
