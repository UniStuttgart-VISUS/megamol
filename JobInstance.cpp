/*
 * JobInstance.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "JobInstance.h"
#include "Module.h"
#include "ModuleNamespace.h"
#include "vislib/sys/AutoLock.h"
#include "vislib/sys/Log.h"

using namespace megamol::core;


/*
 * JobInstance::JobInstance
 */
JobInstance::JobInstance(void) : ModuleNamespace(""), ApiHandle(), job(NULL) {
    // intentionally empty
}


/*
 * JobInstance::~JobInstance
 */
JobInstance::~JobInstance(void) {
    this->Terminate();
    this->job = NULL; // DO NOT DELETE
}


/*
 * JobInstance::Initialize
 */
bool JobInstance::Initialize(ModuleNamespace *ns, job::AbstractJob *job) {
    if ((this->job != NULL) || (ns == NULL) || (job == NULL)) {
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

    this->job = job;
    // Job must be started AFTER setting the parameter values
    //if (!this->job->Start()) {
    //    return false;
    //}

    return true;
}


/*
 * JobInstance::Terminate
 */
void JobInstance::Terminate(void) {
    if (this->job != NULL) {
        this->job->Terminate();
    }
    this->job = NULL; // DO NOT DELETE
}


/*
 * JobInstance::ClearCleanupMark
 */
void JobInstance::ClearCleanupMark(void) {
    if (!this->CleanupMark()) return;

    ModuleNamespace::ClearCleanupMark();
    Module *jobMod = dynamic_cast<Module*>(this->job);
    if (jobMod != NULL) {
        jobMod->ClearCleanupMark();
    }
}


/*
 * JobInstance::PerformCleanup
 */
void JobInstance::PerformCleanup(void) {
    if (this->CleanupMark()) {
        // this should never happen!
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN,
            "Internal Error: JobInstance marked for cleanup.\n");
    }
    ModuleNamespace::PerformCleanup();
}
