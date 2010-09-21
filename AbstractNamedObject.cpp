/*
 * AbstractNamedObject.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "AbstractNamedObject.h"
#include "vislib/assert.h"
#include "vislib/AutoLock.h"
#include "vislib/Log.h"
#include "vislib/StackTrace.h"
#include "vislib/UnsupportedOperationException.h"

using namespace megamol::core;

/****************************************************************************/


/*
 * AbstractNamedObject::GraphLocker::GraphLocker
 */
AbstractNamedObject::GraphLocker::GraphLocker(const AbstractNamedObject *obj,
        bool writelock) : vislib::sys::SyncObject(),
        writelock(writelock), root(NULL) {
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
    VLSTACKTRACE("GraphLocker::Lock", __FILE__, __LINE__);
    this->root->LockModuleGraph(this->writelock);
}


/*
 * AbstractNamedObject::GraphLocker::Unlock
 */
void AbstractNamedObject::GraphLocker::Unlock(void) {
    VLSTACKTRACE("GraphLocker::Unlock", __FILE__, __LINE__);
    this->root->UnlockModuleGraph();
}

/****************************************************************************/


/*
 * AbstractNamedObject::~AbstractNamedObject
 */
AbstractNamedObject::~AbstractNamedObject(void) {
    this->parent = NULL; // DO NOT DELETE
    this->owner = NULL; // DO NOT DELETE
}


/*
 * AbstractNamedObject::FullName
 */
vislib::StringA AbstractNamedObject::FullName(void) const {
    AbstractNamedObject::GraphLocker locker(this, false);
    vislib::sys::AutoLock lock(locker);
    vislib::StringA name;
    const AbstractNamedObject *ano = this;
    while (ano != NULL) {
        if (ano->Name().IsEmpty() && (ano->Parent() == NULL)) {
            break;
        }
        name.Prepend(ano->Name());
        name.Prepend("::");
        ano = ano->Parent();
    }
    return name;
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
        //vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO + 850,
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
bool AbstractNamedObject::IsParamRelevant(
        vislib::SingleLinkedList<const AbstractNamedObject*>& searched,
        const vislib::SmartPtr<param::AbstractParam>& param) const {
    throw vislib::UnsupportedOperationException(
        "AbstractNamedObject::IsParamRelevant", __FILE__, __LINE__);
}


/*
 * AbstractNamedObject::LockModuleGraph
 */
void AbstractNamedObject::LockModuleGraph(bool write) {
    if (this->parent != NULL) {
        this->RootModule()->LockModuleGraph(write);
    }
}


/*
 * AbstractNamedObject::UnlockModuleGraph
 */
void AbstractNamedObject::UnlockModuleGraph(void) {
    if (this->parent != NULL) {
        this->RootModule()->UnlockModuleGraph();
    }
}


/*
 * AbstractNamedObject::LockModuleGraph
 */
void AbstractNamedObject::LockModuleGraph(bool write) const {
    const_cast<AbstractNamedObject*>(this->RootModule())->LockModuleGraph(write);
}


/*
 * AbstractNamedObject::UnlockModuleGraph
 */
void AbstractNamedObject::UnlockModuleGraph(void) const {
    const_cast<AbstractNamedObject*>(this->RootModule())->UnlockModuleGraph();
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
AbstractNamedObject::AbstractNamedObject(void) : name(),
        parent(NULL), owner(NULL), cleanupMark(false) {
    // intentionally empty
}


/*
 * AbstractNamedObject::SetOwner
 */
void AbstractNamedObject::SetOwner(void *owner) {
    ASSERT(this->owner == NULL);
    this->owner = owner;
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
void AbstractNamedObject::setParent(AbstractNamedObject *parent) {
    this->parent = parent;
}
