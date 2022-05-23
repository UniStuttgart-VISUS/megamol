/*
 * JobInstanceRequest.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "mmcore/JobInstanceRequest.h"


/*
 * megamol::core::JobInstanceRequest::JobInstanceRequest
 */
megamol::core::JobInstanceRequest::JobInstanceRequest(void) : InstanceRequest(), desc(NULL) {
    // intentionally empty
}


/*
 * megamol::core::JobInstanceRequest::JobInstanceRequest
 */
megamol::core::JobInstanceRequest::JobInstanceRequest(const megamol::core::JobInstanceRequest& src)
        : InstanceRequest()
        , desc(NULL) {
    *this = src;
}


/*
 * megamol::core::JobInstanceRequest::~JobInstanceRequest
 */
megamol::core::JobInstanceRequest::~JobInstanceRequest(void) {
    this->desc = NULL; // DO NOT DELETE
}


/*
 * megamol::core::JobInstanceRequest::operator=
 */
megamol::core::JobInstanceRequest& megamol::core::JobInstanceRequest::operator=(
    const megamol::core::JobInstanceRequest& rhs) {
    if (&rhs == this)
        return *this;
    ParamValueSetRequest::operator=(rhs);
    this->SetName(rhs.Name());
    this->desc = rhs.desc;
    return *this;
}


/*
 * megamol::core::JobInstanceRequest::operator==
 */
bool megamol::core::JobInstanceRequest::operator==(const megamol::core::JobInstanceRequest& rhs) const {
    return ParamValueSetRequest::operator==(rhs) && (this->Name() == rhs.Name()) && (this->desc == rhs.desc);
}
