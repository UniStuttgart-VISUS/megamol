/*
 * ViewInstanceRequest.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "mmcore/ViewInstanceRequest.h"


/*
 * megamol::core::ViewInstanceRequest::ViewInstanceRequest
 */
megamol::core::ViewInstanceRequest::ViewInstanceRequest(void) : InstanceRequest(), desc(NULL) {
    // intentionally empty
}


/*
 * megamol::core::ViewInstanceRequest::ViewInstanceRequest
 */
megamol::core::ViewInstanceRequest::ViewInstanceRequest(const megamol::core::ViewInstanceRequest& src)
        : InstanceRequest()
        , desc(NULL) {
    *this = src;
}


/*
 * megamol::core::ViewInstanceRequest::~ViewInstanceRequest
 */
megamol::core::ViewInstanceRequest::~ViewInstanceRequest(void) {
    this->desc = NULL; // DO NOT DELETE
}


/*
 * megamol::core::ViewInstanceRequest::operator=
 */
megamol::core::ViewInstanceRequest& megamol::core::ViewInstanceRequest::operator=(
    const megamol::core::ViewInstanceRequest& rhs) {
    if (&rhs == this)
        return *this;
    ParamValueSetRequest::operator=(rhs);
    this->SetName(rhs.Name());
    this->desc = rhs.desc;
    return *this;
}


/*
 * megamol::core::ViewInstanceRequest::operator==
 */
bool megamol::core::ViewInstanceRequest::operator==(const megamol::core::ViewInstanceRequest& rhs) const {
    return ParamValueSetRequest::operator==(rhs) && (this->Name() == rhs.Name()) && (this->desc == rhs.desc);
}
