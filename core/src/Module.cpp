/*
 * Module.cpp
 *
 * Copyright (C) 2009-2015 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/RigRendering.h"
#include "mmcore/Module.h"
#include "mmcore/AbstractSlot.h"
#include "mmcore/CoreInstance.h"
#include <typeinfo>
#include "vislib/assert.h"
#include "vislib/sys/AutoLock.h"
#include "vislib/IllegalParamException.h"
#include "vislib/IllegalStateException.h"
#include "mmcore/utility/log/Log.h"

#ifdef RIG_RENDERCALLS_WITH_DEBUGGROUPS
#include "mmcore/view/Renderer2DModule.h"
#include "mmcore/view/Renderer3DModule.h"
#include "mmcore/view/Renderer3DModuleGL.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#endif

using namespace megamol::core;


/*
 * Module::Module
 */
Module::Module(void) : AbstractNamedObjectContainer(), created(false) {
    // intentionally empty ATM
}


/*
 * Module::~Module
 */
Module::~Module(void) {
    if (this->created == true) {
        throw vislib::IllegalStateException(
            "You must release all resources in the proper derived dtor.",
            __FILE__, __LINE__);
    }
}


/*
 * Module::Create
 */
bool Module::Create(std::vector<megamol::frontend::FrontendResource> resources) {
    using megamol::core::utility::log::Log;

    this->frontend_resources = resources;

    ASSERT(this->instance() != NULL);
    if (!this->created) {
#ifdef RIG_RENDERCALLS_WITH_DEBUGGROUPS
        auto p3 = dynamic_cast<core::view::Renderer3DModule*>(this);
        auto p3_2 = dynamic_cast<core::view::Renderer3DModuleGL*>(this);
        auto p2 = dynamic_cast<core::view::Renderer2DModule*>(this);
        if (p2 || p3 || p3_2) {
            std::string output = this->ClassName();
            output += "::create";
            glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 1234, -1, output.c_str());
        }
#endif
        this->created = this->create();
#ifdef RIG_RENDERCALLS_WITH_DEBUGGROUPS
        if (p2 || p3 || p3_2) glPopDebugGroup();
#endif
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO + 350,
            "%s module \"%s\"\n", ((this->created) ? "Created"
            : "Failed to create"), typeid(*this).name());
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
AbstractSlot * Module::FindSlot(const vislib::StringA& name) {
    child_list_type::iterator iter, end;
    iter = this->ChildList_Begin();
    end = this->ChildList_End();
    for (; iter != end; ++iter) {
        AbstractSlot* slot = dynamic_cast<AbstractSlot*>(iter->get());
        if (slot == NULL) continue;
        if (slot->Name().Equals(name, false)) {
            return slot;
        }
    }
    return NULL;
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
void Module::PerformCleanup(void) {
    // Do not proceed into the children, because they will be deleted
    // automatically
    AbstractNamedObject::PerformCleanup();

    if (!this->CleanupMark()) return;

    // clear list of children
    child_list_type::iterator b, e;
    while(true) {
        b = this->ChildList_Begin();
        e = this->ChildList_End();
        if (b == e) break;
        this->removeChild(*b);
    }

}


/*
 * Module::getRelevantConfigValue
 */
vislib::StringA Module::getRelevantConfigValue(vislib::StringA name) {
    vislib::StringA ret = vislib::StringA::EMPTY;
    const utility::Configuration& cfg = this->GetCoreInstance()->Configuration();
    vislib::StringA drn = this->GetDemiRootName();
    vislib::StringA test = drn;
    test.Append("-");
    test.Append(name);
    vislib::StringA test2("*-");
    test2.Append(name);
    if (cfg.IsConfigValueSet(test)) {
        ret = cfg.ConfigValue(test);
    } else if (cfg.IsConfigValueSet(test2)) {
        ret = cfg.ConfigValue(test2);
    } else if (cfg.IsConfigValueSet(name)) {
        ret = cfg.ConfigValue(name);
    }

    return ret;
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
    this->addChild(::std::shared_ptr<AbstractNamedObject>(slot, [](AbstractNamedObject* d){}));
    slot->SetOwner(this);
    slot->MakeAvailable();
}

void Module::SetSlotUnavailable(AbstractSlot *slot) {
    if (slot == NULL) {
        throw vislib::IllegalParamException("slot", __FILE__, __LINE__);
    }
    if (slot->GetStatus() == AbstractSlot::STATUS_CONNECTED) {
        throw vislib::IllegalStateException("slot", __FILE__, __LINE__);
    }
    if (slot->GetStatus() == AbstractSlot::STATUS_UNAVAILABLE) return;

    if (!this->FindSlot(slot->Name()))  {
        throw vislib::IllegalParamException("A slot with this name is not registered", __FILE__, __LINE__);
    }

    this->removeChild(::std::shared_ptr<AbstractNamedObject>(slot, [](AbstractNamedObject* d){}));
    slot->SetOwner(nullptr);
    slot->MakeUnavailable();

}



/*
 * Module::setModuleName
 */
void Module::setModuleName(const vislib::StringA& name) {
    this->setName(name);
}
