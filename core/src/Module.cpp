/**
 * MegaMol
 * Copyright (c) 2009, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/Module.h"

#include <typeinfo>

#include "mmcore/AbstractSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/utility/log/Log.h"
#include "vislib/IllegalParamException.h"
#include "vislib/IllegalStateException.h"
#include "vislib/assert.h"
#include "vislib/sys/AutoLock.h"

using namespace megamol::core;


/*
 * Module::Module
 */
Module::Module() : AbstractNamedObjectContainer(), created(false) {
    // intentionally empty ATM
}


/*
 * Module::~Module
 */
Module::~Module() {
    if (this->created == true) {
        throw vislib::IllegalStateException(
            "You must release all resources in the proper derived dtor.", __FILE__, __LINE__);
    }
}


/*
 * Module::Create
 */
bool Module::Create(std::vector<megamol::frontend::FrontendResource> resources) {
    using megamol::core::utility::log::Log;

    this->frontend_resources = {resources}; // put resources in hash map using type hashes of present resources

    if (!this->created) {
        this->created = this->create();
        Log::DefaultLog.WriteInfo(
            "%s module \"%s\"\n", ((this->created) ? "Created" : "Failed to create"), typeid(*this).name());
    }
    if (this->created) {
        // Now reregister parents at children
        this->fixParentBackreferences();
    }

    return this->created;
}


/*
 * Module::FindSlot
 */
AbstractSlot* Module::FindSlot(const vislib::StringA& name) {
    child_list_type::iterator iter, end;
    iter = this->ChildList_Begin();
    end = this->ChildList_End();
    for (; iter != end; ++iter) {
        AbstractSlot* slot = dynamic_cast<AbstractSlot*>(iter->get());
        if (slot == nullptr)
            continue;
        if (slot->Name().Equals(name, false)) {
            return slot;
        }
    }
    return nullptr;
}


/*
 * Module::GetDemiRootName
 */
vislib::StringA Module::GetDemiRootName() const {
    AbstractNamedObject::const_ptr_type tm = this->shared_from_this();
    while (tm->Parent() && !tm->Parent()->Name().IsEmpty()) {
        tm = tm->Parent();
    }
    return tm->Name();
}


/*
 * Module::Release
 */
void Module::Release(std::vector<megamol::frontend::FrontendResource> resources) {
    using megamol::core::utility::log::Log;

    if (this->created) {
        this->release();
        this->created = false;
        Log::DefaultLog.WriteInfo("Released module \"%s\"\n", typeid(*this).name());
    }
}


/*
 * Module::ClearCleanupMark
 */
void Module::ClearCleanupMark() {
    if (!this->CleanupMark())
        return;

    AbstractNamedObject::ClearCleanupMark();

    child_list_type::iterator iter, end;
    iter = this->ChildList_Begin();
    end = this->ChildList_End();
    for (; iter != end; ++iter) {
        (*iter)->ClearCleanupMark();
    }

    AbstractNamedObject::ptr_type p = this->Parent();
    if (p) {
        p->ClearCleanupMark();
    }
}


/*
 * Module::PerformCleanup
 */
void Module::PerformCleanup() {
    // Do not proceed into the children, because they will be deleted
    // automatically
    AbstractNamedObject::PerformCleanup();

    if (!this->CleanupMark())
        return;

    // clear list of children
    child_list_type::iterator b, e;
    while (true) {
        b = this->ChildList_Begin();
        e = this->ChildList_End();
        if (b == e)
            break;
        this->removeChild(*b);
    }
}


bool Module::AnyParameterDirty() const {
    auto ret = false;
    for (auto it = ChildList_Begin(); it != ChildList_End(); ++it) {
        if (const auto paramSlot = dynamic_cast<param::ParamSlot*>((*it).get())) {
            ret = ret || paramSlot->IsDirty();
        }
    }
    return ret;
}


void Module::ResetAllDirtyFlags() {
    for (auto it = ChildList_Begin(); it != ChildList_End(); ++it) {
        if (const auto paramSlot = dynamic_cast<param::ParamSlot*>((*it).get())) {
            paramSlot->ResetDirty();
        }
    }
}


/*
 * Module::MakeSlotAvailable
 */
void Module::MakeSlotAvailable(AbstractSlot* slot) {
    if (slot == nullptr) {
        throw vislib::IllegalParamException("slot", __FILE__, __LINE__);
    }
    if (slot->GetStatus() != AbstractSlot::STATUS_UNAVAILABLE) {
        throw vislib::IllegalStateException("slot", __FILE__, __LINE__);
    }
    if (this->FindSlot(slot->Name())) {
        throw vislib::IllegalParamException("A slot with this name is already registered", __FILE__, __LINE__);
    }
    this->addChild(::std::shared_ptr<AbstractNamedObject>(slot, [](AbstractNamedObject* d) {}));
    slot->SetOwner(this);
    slot->MakeAvailable();
}

void Module::SetSlotUnavailable(AbstractSlot* slot) {
    if (slot == nullptr) {
        throw vislib::IllegalParamException("slot", __FILE__, __LINE__);
    }
    if (slot->GetStatus() == AbstractSlot::STATUS_CONNECTED) {
        throw vislib::IllegalStateException("slot", __FILE__, __LINE__);
    }
    if (slot->GetStatus() == AbstractSlot::STATUS_UNAVAILABLE)
        return;

    if (!this->FindSlot(slot->Name())) {
        throw vislib::IllegalParamException("A slot with this name is not registered", __FILE__, __LINE__);
    }

    this->removeChild(::std::shared_ptr<AbstractNamedObject>(slot, [](AbstractNamedObject* d) {}));
    slot->SetOwner(nullptr);
    slot->MakeUnavailable();
}


/*
 * Module::setModuleName
 */
void Module::setModuleName(const vislib::StringA& name) {
    this->setName(name);
}
