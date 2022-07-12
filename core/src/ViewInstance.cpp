/*
 * ViewInstance.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "mmcore/ViewInstance.h"
#include "mmcore/Module.h"
#include "mmcore/ModuleNamespace.h"
#include "mmcore/utility/log/Log.h"
#include "vislib/sys/AutoLock.h"

using namespace megamol::core;


/*
 * ViewInstance::ViewInstance
 */
ViewInstance::ViewInstance(void) : ModuleNamespace(""), view(NULL), closeRequestCallback(NULL), closeRequestData(NULL) {
    // intentionally empty
}


/*
 * ViewInstance::~ViewInstance
 */
ViewInstance::~ViewInstance(void) {
    this->Terminate();
    this->view = NULL;                 // DO NOT DELETE
    this->closeRequestCallback = NULL; // DO NOT DELETE
    this->closeRequestData = NULL;     // DO NOT DELETE
}


/*
 * ViewInstance::Initialize
 */
bool ViewInstance::Initialize(ModuleNamespace::ptr_type ns, view::AbstractViewInterface* view) {
    if ((this->view != NULL) || (ns == NULL) || (view == NULL)) {
        return false;
    }

    // this replaces the namespace object ns with this new view instance object

    AbstractNamedObject::GraphLocker locker(ns, true);
    vislib::sys::AutoLock lock(locker);

    ModuleNamespace::ptr_type p = ModuleNamespace::dynamic_pointer_cast(ns->Parent());
    if (!p) {
        return false;
    }

    while (ns->ChildList_Begin() != ns->ChildList_End()) {
        AbstractNamedObject::ptr_type ano = *ns->ChildList_Begin();
        ns->RemoveChild(ano);
        this->AddChild(ano);
    }

    this->setName(ns->Name());

    p->RemoveChild(ns);
    p->AddChild(this->shared_from_this());

    ASSERT(ns->Parent() == NULL);
    ASSERT(ns->ChildList_Begin() == ns->ChildList_End());

    this->view = view;

    return true;
}


/*
 * ViewInstance::ClearCleanupMark
 */
void ViewInstance::ClearCleanupMark(void) {
    if (!this->CleanupMark())
        return;

    ModuleNamespace::ClearCleanupMark();
    Module* viewMod = dynamic_cast<Module*>(this->view);
    if (viewMod != NULL) {
        viewMod->ClearCleanupMark();
    }
}


/*
 * ViewInstance::PerformCleanup
 */
void ViewInstance::PerformCleanup(void) {
    if (this->CleanupMark()) {
        // this should never happen!
        megamol::core::utility::log::Log::DefaultLog.WriteWarn("Internal Error: ViewInstance marked for cleanup.\n");
    }
    ModuleNamespace::PerformCleanup();
}


/*
 * ViewInstance::RequestClose
 */
void ViewInstance::RequestClose(void) {
    if (this->closeRequestCallback != NULL) {
        this->closeRequestCallback(this->closeRequestData);
    }
}
