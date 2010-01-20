/*
 * JobManager.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "JobManager.h"
#include "MegaMolCore.h"
#include "vislib/sysfunctions.h"


/*
 * megamol::console::JobManager::Instance
 */
megamol::console::JobManager*
megamol::console::JobManager::Instance(void) {
    static vislib::SmartPtr<megamol::console::JobManager> inst;
    if (inst.IsNull()) {
        inst = new megamol::console::JobManager();
    }
    return inst.operator->();
}


/*
 * megamol::console::JobManager::JobManager
 */
megamol::console::JobManager::JobManager(void) : jobs() {
}


/*
 * megamol::console::JobManager::~JobManager
 */
megamol::console::JobManager::~JobManager(void) {
}


/*
 * megamol::console::JobManager::Add
 */
void megamol::console::JobManager::Add(
        vislib::SmartPtr<megamol::console::CoreHandle> &job) {
    this->jobs.Add(job);
}


/*
 * megamol::console::JobManager::CheckJobs
 */
bool megamol::console::JobManager::CheckJobs(void) {
    static unsigned int ticks = vislib::sys::GetTicksOfDay();

    // only test once per second!
    if ((vislib::sys::GetTicksOfDay() < ticks) || (ticks + 1000 > vislib::sys::GetTicksOfDay())) {
        return true;
    }

    vislib::SingleLinkedList<vislib::SmartPtr<CoreHandle> >::Iterator iter
        = this->jobs.GetIterator();

    while (iter.HasNext()) {
        vislib::SmartPtr<CoreHandle>& hndl = iter.Next();
        if (!::mmcIsJobRunning(hndl->operator void*())) {
            this->jobs.Remove(iter);
        }
    }

    return !this->jobs.IsEmpty();
}


/*
 * megamol::console::JobManager::TerminateJobs
 */
void megamol::console::JobManager::TerminateJobs(void) {
    vislib::SingleLinkedList<vislib::SmartPtr<CoreHandle> >::Iterator iter
        = this->jobs.GetIterator();

    while (iter.HasNext()) {
        vislib::SmartPtr<CoreHandle>& hndl = iter.Next();
        if (::mmcIsJobRunning(hndl->operator void*())) {
            ::mmcTerminateJob(hndl->operator void*());
        }
    }
}
