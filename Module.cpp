/*
 * Module.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "Module.h"
#include "AbstractSlot.h"
#include "CoreInstance.h"
#include <typeinfo>
#include "vislib/assert.h"
#include "vislib/AutoLock.h"
#include "vislib/IllegalParamException.h"
#include "vislib/IllegalStateException.h"
#include "vislib/Log.h"

using namespace megamol::core;


/*
 * Module::Module
 */
Module::Module(void) : AbstractNamedObjectContainer(), coreInst(NULL),
        created(false) {
    // intentionally empty ATM
}


/*
 * Module::~Module
 */
Module::~Module(void) {
    this->coreInst = NULL; // DO NOT DELETE
    if (this->created == true) {
        throw vislib::IllegalStateException(
            "You must release all resources in the proper derived dtor.",
            __FILE__, __LINE__);
    }
}


/*
 * Module::Create
 */
bool Module::Create(void) {
    using vislib::sys::Log;
    ASSERT(this->instance() != NULL);
    if (!this->created) {
        this->created = this->create();
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO + 350,
            "%s module \"%s\"\n", ((this->created) ? "Created"
            : "Failed to create"), typeid(*this).name());
    }
    return this->created;
}


/*
 * Module::FindSlot
 */
AbstractSlot * Module::FindSlot(const vislib::StringA& name) {
    ChildList::Iterator iter = this->getChildIterator();
    while (iter.HasNext()) {
        AbstractSlot* slot = dynamic_cast<AbstractSlot*>(iter.Next());
        if (slot == NULL) continue;
        if (slot->Name().Equals(name, false)) {
            return slot;
        }
    }
    return NULL;
}


/*
 * Module::Release
 */
void Module::Release(void) {
    using vislib::sys::Log;
    if (this->created) {
        this->release();
        this->created = false;
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO + 350,
            "Released module \"%s\"\n", typeid(*this).name());
    }
}


/*
 * Module::ClearCleanupMark
 */
void Module::ClearCleanupMark(void) {
    if (!this->CleanupMark()) return;

    AbstractNamedObject::ClearCleanupMark();
    ChildList::Iterator iter = this->GetChildIterator();
    while (iter.HasNext()) {
        iter.Next()->ClearCleanupMark();
    }

    if (this->Parent() != NULL){
        this->Parent()->ClearCleanupMark();
    }
}


/*
 * Module::PerformCleanup
 */
void Module::PerformCleanup(void) {
    // Do not proceed into the children, because they will be deleted
    // automatically
    AbstractNamedObject::PerformCleanup();

    if (!this->CleanupMark()) return;

    ChildList::Iterator iter;
    AbstractNamedObject *c;

    // just remove the pointers, so nobody gets confused
    do {
        iter = this->getChildIterator();
        if (iter.HasNext()) {
            c = iter.Next();
            this->removeChild(c);
        } else {
            c = NULL;
        }
    } while (c != NULL);

}


/*
 * Module::MakeSlotAvailable
 */
void Module::MakeSlotAvailable(AbstractSlot *slot) {
    if (slot == NULL) {
        throw vislib::IllegalParamException("slot", __FILE__, __LINE__);
    }
    if (slot->GetStatus() != AbstractSlot::STATUS_UNAVAILABLE) {
        throw vislib::IllegalStateException("slot", __FILE__, __LINE__);
    }
    if (this->FindSlot(slot->Name()))  {
        throw vislib::IllegalParamException(
            "A slot with this name is already registered",
            __FILE__, __LINE__);
    }
    this->addChild(slot);
    slot->SetOwner(this);
    slot->MakeAvailable();
}


/*
 * Module::setModuleName
 */
void Module::setModuleName(const vislib::StringA& name) {
    this->setName(name);
}
