/*
 * AbstractSlot.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "mmcore/AbstractSlot.h"
#include "vislib/IllegalStateException.h"
#include "vislib/UnsupportedOperationException.h"

using namespace megamol::core;


/*
 * AbstractSlot::~AbstractSlot
 */
AbstractSlot::~AbstractSlot(void) {
    this->listeners.Clear(); // DO NOT DELETE ELEMENTS
}


/*
 * AbstractSlot::MakeAvailable
 */
void AbstractSlot::MakeAvailable(void) {
    if (this->status != STATUS_UNAVAILABLE) {
        throw vislib::IllegalStateException(
            "Status of slot is illegal for this operation",
            __FILE__, __LINE__);
    }
    this->status = STATUS_ENABLED;
}


/*
 * AbstractSlot::MakeUnavailable
 */
void AbstractSlot::MakeUnavailable(void) {
    if (this->status != STATUS_ENABLED) {
        throw vislib::IllegalStateException(
            "Status of slot is illegal for this operation",
            __FILE__, __LINE__);
    }
    this->status = STATUS_UNAVAILABLE;
}


/*
 * AbstractSlot::IsParamRelevant
 */
bool AbstractSlot::IsParamRelevant(
        vislib::SingleLinkedList<const AbstractNamedObject*>& searched,
        const vislib::SmartPtr<param::AbstractParam>& param) const {
    return false;
}


/*
 * AbstractSlot::AbstractSlot
 */
AbstractSlot::AbstractSlot(const vislib::StringA& name, const vislib::StringA& desc)
    : AbstractNamedObject()
    , desc(desc)
    , status(STATUS_UNAVAILABLE)
    , connectedRefCnt(0)
    , listeners()
{
    this->setName(name);
}


/*
 * AbstractSlot::SetStatusConnected
 */
void AbstractSlot::SetStatusConnected(bool connected) {
    if (this->status == STATUS_UNAVAILABLE) {
        throw vislib::IllegalStateException(
            "Status of slot is illegal for this operation",
            __FILE__, __LINE__);
    }

    if (connected) {
        this->connectedRefCnt++;
        this->status = STATUS_CONNECTED;
    } else {
        this->connectedRefCnt--;
        this->status = this->connectedRefCnt
            ? STATUS_CONNECTED : STATUS_ENABLED;
    }

    vislib::SingleLinkedList<Listener*>::Iterator i
        = this->listeners.GetIterator();
    while (i.HasNext()) {
        Listener *l = i.Next();
        if (connected) {
            l->OnConnect(*this);
        } else {
            l->OnDisconnect(*this);
        }
    }
}


/*
 * AbstractSlot::AbstractSlot
 */
AbstractSlot::AbstractSlot(const AbstractSlot& src) {
    throw vislib::UnsupportedOperationException("Copy Ctor",
        __FILE__, __LINE__);
}


/*
 * AbstractSlot::operator=
 */
AbstractSlot& AbstractSlot::operator=(const AbstractSlot& rhs) {
    if (this != &rhs) {
        throw vislib::UnsupportedOperationException("operator=",
            __FILE__, __LINE__);
    }
    return *this;
}
