/*
 * ParamValueSetRequest.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "mmcore/ParamValueSetRequest.h"


/*
 * megamol::core::ParamValueSetRequest::ParamValueSetRequest
 */
megamol::core::ParamValueSetRequest::ParamValueSetRequest(void) : paramValues() {}


/*
 * megamol::core::ParamValueSetRequest::ParamValueSetRequest
 */
megamol::core::ParamValueSetRequest::ParamValueSetRequest(const megamol::core::ParamValueSetRequest& src)
        : paramValues() {
    *this = src;
}


/*
 * megamol::core::ParamValueSetRequest::~ParamValueSetRequest
 */
megamol::core::ParamValueSetRequest::~ParamValueSetRequest(void) {
    this->paramValues.Clear();
}


/*
 * megamol::core::ParamValueSetRequest::operator=
 */
megamol::core::ParamValueSetRequest& megamol::core::ParamValueSetRequest::operator=(
    const megamol::core::ParamValueSetRequest& rhs) {
    if (&rhs == this)
        return *this;
    this->paramValues = rhs.paramValues;
    return *this;
}


/*
 * megamol::core::ParamValueSetRequest::operator==
 */
bool megamol::core::ParamValueSetRequest::operator==(const megamol::core::ParamValueSetRequest& rhs) const {
    return this->paramValues == rhs.paramValues;
}
