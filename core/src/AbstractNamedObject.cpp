/*
 * AbstractNamedObject.cpp
 *
 * Copyright (C) 2009-2015 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#include "mmcore/AbstractNamedObject.h"
#include "mmcore/utility/log/Log.h"
#include "stdafx.h"
#include "vislib/UnsupportedOperationException.h"
#include "vislib/assert.h"
#include "vislib/sys/AutoLock.h"

using namespace megamol::core;

/****************************************************************************/


/*
 * AbstractNamedObject::GraphLocker::GraphLocker
 */
AbstractNamedObject::GraphLocker::GraphLocker(AbstractNamedObject::const_ptr_type obj, bool writelock)
        : vislib::sys::SyncObject()
        , writelock(writelock)
        , root(NULL) {
    ASSERT(obj != NULL);

    this->root = obj->RootModule();
}


/*
 * AbstractNamedObject::GraphLocker::~GraphLocker
 */
AbstractNamedObject::GraphLocker::~GraphLocker(void) {
    this->root = NULL; // DO NOT DELETE
}


/*
 * AbstractNamedObject::GraphLocker::Lock
 */
void AbstractNamedObject::GraphLocker::Lock(void) {
    //if (this->writelock) {
    //        this->root->ModuleGraphLock().LockExclusive();
    //} else {
    //    this->root->ModuleGraphLock().LockExclusive();
    //}
}


/*
 * AbstractNamedObject::GraphLocker::Unlock
 */
void AbstractNamedObject::GraphLocker::Unlock(void) {
    //if (this->writelock) {
    //        this->root->ModuleGraphLock().UnlockExclusive();
    //} else {
    //    this->root->ModuleGraphLock().UnlockExclusive();
    //}
}

/****************************************************************************/


/*
 * AbstractNamedObject::~AbstractNamedObject
 */
AbstractNamedObject::~AbstractNamedObject(void) {
    this->parent.reset();
    this->owner = nullptr; // DO NOT DELETE
}


/*
 * AbstractNamedObject::FullName
 */
vislib::StringA AbstractNamedObject::FullName(void) const {
    try {
        AbstractNamedObject::GraphLocker locker(this->shared_from_this(), false);
        vislib::sys::AutoLock lock(locker);
        vislib::StringA name;
        const_ptr_type ano = this->shared_from_this();
        while (ano) {
            if (ano->Name().IsEmpty() && (!ano->Parent())) {
                break;
            }
            name.Prepend(ano->Name());
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
void AbstractNamedObject::SetAllCleanupMarks(void) {
    this->cleanupMark = true;
}


/*
 * AbstractNamedObject::ClearCleanupMark
 */
void AbstractNamedObject::ClearCleanupMark(void) {
    this->cleanupMark = false;
}


/*
 * AbstractNamedObject::PerformCleanup
 */
void AbstractNamedObject::PerformCleanup(void) {
    if (this->cleanupMark) {
        // message removed because of quickstart module peeking
        //megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_INFO + 850,
        //    "Module \"%s\" marked for cleanup\n", this->Name().PeekBuffer());
    }

    // intentionally empty
}


/*
 * AbstractNamedObject::DisconnectCalls
 */
void AbstractNamedObject::DisconnectCalls(void) {
    // intentionally empty
}


/*
 * AbstractNamedObject::IsParamRelevant
 */
bool AbstractNamedObject::IsParamRelevant(vislib::SingleLinkedList<const AbstractNamedObject*>& searched,
    const vislib::SmartPtr<param::AbstractParam>& param) const {
    throw vislib::UnsupportedOperationException("AbstractNamedObject::IsParamRelevant", __FILE__, __LINE__);
}


/*
 * AbstractNamedObject::ModuleGraphLock
 */
vislib::sys::AbstractReaderWriterLock& AbstractNamedObject::ModuleGraphLock(void) {
    ASSERT(!this->parent.expired()); // HAZARD: better return a dummy object
    return this->RootModule()->ModuleGraphLock();
}


/*
 * AbstractNamedObject::ModuleGraphLock
 */
vislib::sys::AbstractReaderWriterLock& AbstractNamedObject::ModuleGraphLock(void) const {
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
AbstractNamedObject::AbstractNamedObject(void)
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
