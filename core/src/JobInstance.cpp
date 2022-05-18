/*
 * JobInstance.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"

#include "mmcore/JobInstance.h"
#include "mmcore/Module.h"
#include "mmcore/ModuleNamespace.h"
#include "mmcore/utility/log/Log.h"
#include "vislib/sys/AutoLock.h"

using namespace megamol::core;


/*
 * JobInstance::JobInstance
 */
JobInstance::JobInstance(void) : ModuleNamespace(""), job(NULL) {
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
bool JobInstance::Initialize(ModuleNamespace::ptr_type ns, job::AbstractJob* job) {
    if ((this->job != NULL) || (ns == NULL) || (job == NULL)) {
        return false;
    }

    AbstractNamedObject::GraphLocker locker(ns, true);
    vislib::sys::AutoLock lock(locker);

    ModuleNamespace::ptr_type p = ModuleNamespace::dynamic_pointer_cast(ns->Parent());
    if (p == NULL) {
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
    if (!this->CleanupMark())
        return;

    ModuleNamespace::ClearCleanupMark();
    Module* jobMod = dynamic_cast<Module*>(this->job);
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
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(
            megamol::core::utility::log::Log::LEVEL_WARN, "Internal Error: JobInstance marked for cleanup.\n");
    }
    ModuleNamespace::PerformCleanup();
}
