/*
 * ReferenceCounted.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/ReferenceCounted.h"


#include "vislib/assert.h"
#include "vislib/Trace.h"


/*
 * vislib::ReferenceCounted::AddRef
 */
UINT32 vislib::ReferenceCounted::AddRef(void) {
    VLTRACE(Trace::LEVEL_VL_ANNOYINGLY_VERBOSE, "Add reference to 0x%p, "
        "reference count is now %u.\n", this, this->cntRefs + 1);
    return ++this->cntRefs;
}

/*
 * vislib::ReferenceCounted::Release
 */
UINT32 vislib::ReferenceCounted::Release(void) {
    ASSERT(this->cntRefs > 0);
    UINT32 retval = --this->cntRefs;
    VLTRACE(Trace::LEVEL_VL_ANNOYINGLY_VERBOSE, "Released object 0x%p, "
        "reference count is now %u.\n", this, this->cntRefs);
    if (this->cntRefs == 0) {
        delete this;
    }
    return retval;
}


/*
 * vislib::ReferenceCounted::ReferenceCounted
 */
vislib::ReferenceCounted::ReferenceCounted(void) : cntRefs(1) {
    VLTRACE(Trace::LEVEL_VL_ANNOYINGLY_VERBOSE, "Object 0x%p initialised, "
        "reference count is now %u.\n", this, this->cntRefs);
}


/*
 * ReferenceCounted::ReferenceCounted
 */
vislib::ReferenceCounted::ReferenceCounted(const ReferenceCounted& rhs) 
        : cntRefs(1) {
}


/*
 * vislib::ReferenceCounted::~ReferenceCounted
 */
vislib::ReferenceCounted::~ReferenceCounted(void) {
    ASSERT(this->cntRefs == 0);
}


/*
 * vislib::ReferenceCounted::operator =
 */
vislib::ReferenceCounted& vislib::ReferenceCounted::operator =(
        const ReferenceCounted& rhs) {
    // Nothing to be done! No not modify the reference count!
    return *this;
}
