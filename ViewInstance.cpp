/*
 * ViewInstance.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "ViewInstance.h"
#include "Module.h"
#include "ModuleNamespace.h"
#include "vislib/sys/AutoLock.h"
#include "vislib/sys/Log.h"
#include "vislib/StackTrace.h"

using namespace megamol::core;


/*
 * ViewInstance::ViewInstance
 */
ViewInstance::ViewInstance(void) : ModuleNamespace(""), ApiHandle(),
        view(NULL), closeRequestCallback(NULL), closeRequestData(NULL) {
    // intentionally empty
}


/*
 * ViewInstance::~ViewInstance
 */
ViewInstance::~ViewInstance(void) {
    this->Terminate();
    this->view = NULL; // DO NOT DELETE
    this->closeRequestCallback = NULL; // DO NOT DELETE
    this->closeRequestData = NULL; // DO NOT DELETE
}


/*
 * ViewInstance::Initialize
 */
bool ViewInstance::Initialize(ModuleNamespace *ns, view::AbstractView *view) {
    VLSTACKTRACE("Initialize", __FILE__, __LINE__);
    if ((this->view != NULL) || (ns == NULL) || (view == NULL)) {
        return false;
    }

    AbstractNamedObject::GraphLocker locker(ns, true);
    vislib::sys::AutoLock lock(locker);

    ModuleNamespace *p = dynamic_cast<ModuleNamespace*>(ns->Parent());
    if (p == NULL) {
        return false;
    }

    AbstractNamedObjectContainer::ChildList::Iterator iter = ns->GetChildIterator();
    while (iter.HasNext()) {
        AbstractNamedObject *ano = iter.Next();
        ns->RemoveChild(ano);
        this->AddChild(ano);
    }

    this->setName(ns->Name());

    p->RemoveChild(ns);
    p->AddChild(this);

    ASSERT(ns->Parent() == NULL);
    ASSERT(!ns->GetChildIterator().HasNext());

    this->view = view;

    return true;
}


/*
 * ViewInstance::ClearCleanupMark
 */
void ViewInstance::ClearCleanupMark(void) {
    if (!this->CleanupMark()) return;

    ModuleNamespace::ClearCleanupMark();
    Module *viewMod = dynamic_cast<Module*>(this->view);
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
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN,
            "Internal Error: ViewInstance marked for cleanup.\n");
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
