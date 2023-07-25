/**
 * MegaMol
 * Copyright (c) 2009, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/AbstractNamedObject.h"
#include "mmcore/utility/log/Log.h"
#include "vislib/UnsupportedOperationException.h"
#include "vislib/assert.h"
#include "vislib/sys/AutoLock.h"

using namespace megamol::core;

/****************************************************************************/


/*
 * AbstractNamedObject::~AbstractNamedObject
 */
AbstractNamedObject::~AbstractNamedObject() {
    this->parent.reset();
    this->owner = nullptr; // DO NOT DELETE
}


/*
 * AbstractNamedObject::FullName
 */
vislib::StringA AbstractNamedObject::FullName() const {
    try {
        vislib::StringA name;
        const_ptr_type ano = this->shared_from_this();
        while (ano) {
            if (ano->Name().IsEmpty() && (!ano->Parent())) {
                break;
            }
            name.Prepend(ano->Name());
            if (!name.StartsWith("::", false))
                name.Prepend("::");
            ano = ano->Parent();
        }
        return name;
    } catch (...) { // evil multi-threading and broken shared ownership results in ill behaviour
        return "";
    }
}


/*
 * AbstractNamedObject::SetAllCleanupMarks
 */
void AbstractNamedObject::SetAllCleanupMarks() {
    this->cleanupMark = true;
}


/*
 * AbstractNamedObject::ClearCleanupMark
 */
void AbstractNamedObject::ClearCleanupMark() {
    this->cleanupMark = false;
}


/*
 * AbstractNamedObject::PerformCleanup
 */
void AbstractNamedObject::PerformCleanup() {
    if (this->cleanupMark) {
        // message removed because of quickstart module peeking
        //megamol::core::utility::log::Log::DefaultLog.WriteInfo(
        //    "Module \"%s\" marked for cleanup\n", this->Name().PeekBuffer());
    }

    // intentionally empty
}


/*
 * AbstractNamedObject::DisconnectCalls
 */
void AbstractNamedObject::DisconnectCalls() {
    // intentionally empty
}


/*
 * AbstractNamedObject::ModuleGraphLock
 */
vislib::sys::AbstractReaderWriterLock& AbstractNamedObject::ModuleGraphLock() {
    ASSERT(!this->parent.expired()); // HAZARD: better return a dummy object
    return this->RootModule()->ModuleGraphLock();
}


/*
 * AbstractNamedObject::ModuleGraphLock
 */
vislib::sys::AbstractReaderWriterLock& AbstractNamedObject::ModuleGraphLock() const {
    ASSERT(!this->parent.expired()); // HAZARD: better return a dummy object
    return this->RootModule()->ModuleGraphLock();
}


/*
 * AbstractNamedObject::isNameValid
 */
bool AbstractNamedObject::isNameValid(const vislib::StringA& name) {
    return name.Find(':') == vislib::StringA::INVALID_POS;
}


/*
 * AbstractNamedObject::AbstractNamedObject
 */
AbstractNamedObject::AbstractNamedObject()
        : enable_shared_from_this()
        , name()
        , parent()
        , owner(nullptr)
        , cleanupMark(false) {
    // intentionally empty
}


/*
 * AbstractNamedObject::SetOwner
 */
void AbstractNamedObject::SetOwner(void* owner) {
    if (owner == nullptr) {
        this->owner = nullptr;
    } else {
        ASSERT(this->owner == nullptr);
        this->owner = owner;
    }
}


/*
 * AbstractNamedObject::setName
 */
void AbstractNamedObject::setName(const vislib::StringA& name) {
    this->name = name;
}


/*
 * AbstractNamedObject::setParent
 */
void AbstractNamedObject::setParent(AbstractNamedObject::weak_ptr_type parent) {
    this->parent = parent;
}
