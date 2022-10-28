/*
 * Module.cpp
 *
 * Copyright (C) 2009-2015 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "mmcore/Module.h"

#include <typeinfo>

#include "mmcore/AbstractSlot.h"
#include "mmcore/utility/log/Log.h"
#include "vislib/IllegalParamException.h"
#include "vislib/IllegalStateException.h"
#include "vislib/assert.h"
#include "vislib/sys/AutoLock.h"

#ifdef RIG_RENDERCALLS_WITH_DEBUGGROUPS
#include "mmcore/view/Renderer2DModule.h"
#include "mmcore/view/Renderer3DModule.h"
#include "mmstd_gl/renderer/Renderer3DModuleGL.h"
#include "vislib_gl/graphics/gl/IncludeAllGL.h"
#endif

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
#ifdef RIG_RENDERCALLS_WITH_DEBUGGROUPS
        auto p3 = dynamic_cast<core::view::Renderer3DModule*>(this);
        auto p3_2 = dynamic_cast<mmstd_gl::Renderer3DModuleGL*>(this);
        auto p2 = dynamic_cast<core::view::Renderer2DModule*>(this);
        if (p2 || p3 || p3_2) {
            std::string output = this->ClassName();
            output += "::create";
            glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 1234, -1, output.c_str());
        }
#endif
        this->created = this->create();
#ifdef RIG_RENDERCALLS_WITH_DEBUGGROUPS
        if (p2 || p3 || p3_2)
            glPopDebugGroup();
#endif
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
