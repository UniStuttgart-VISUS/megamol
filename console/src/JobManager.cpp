/*
 * JobManager.cpp
 *
 * Copyright (C) 2008, 2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "JobManager.h"
#include "mmcore/api/MegaMolCore.h"
#include <cassert>
#include <algorithm>
#include <chrono>
//#include "vislib/sys/sysfunctions.h"


/*
 * megamol::console::JobManager::Instance
 */
megamol::console::JobManager& megamol::console::JobManager::Instance(void) {
    static megamol::console::JobManager inst;
    return inst;
}

/*
 * megamol::console::JobManager::~JobManager
 */
megamol::console::JobManager::~JobManager(void) {
    assert(jobs.empty());
}

/*
 * megamol::console::JobManager::JobManager
 */
megamol::console::JobManager::JobManager(void) : jobs(), terminating(false) {
    // intentionally empty
}

bool megamol::console::JobManager::IsAlive(void) const {
    return !jobs.empty() && !terminating;
}

void megamol::console::JobManager::Update(bool force) {
    if (jobs.empty()) return;

    static std::chrono::system_clock::time_point lastUpdate;
    std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    if (!force && (std::chrono::duration_cast<std::chrono::seconds>(now - lastUpdate).count() <= 0)) return;
    lastUpdate = now;

    bool cleaning = false;

    for (std::shared_ptr<Job> j : jobs) {
        if (!j->started && !terminating) j->Start();
        if (!j->IsRunning()) cleaning = true;
    }

    if (!cleaning) return;

    std::vector<std::shared_ptr<Job> >::iterator e = jobs.end();
    for (std::vector<std::shared_ptr<Job> >::iterator i = jobs.begin(); i != e; ) {
        if ((*i)->IsRunning()) ++i;
        else {
            i = jobs.erase(i);
            e = jobs.end(); // because we potentially changed everything.
        }
    }

}

void megamol::console::JobManager::Shutdown(void) {
    for (std::shared_ptr<Job> j : jobs) {
        if (j->IsRunning()) ::mmcTerminateJob(j->hJob);
    }
    terminating = true;
    while (!jobs.empty()) Update(true);
}

bool megamol::console::JobManager::InstantiatePendingJob(void *hCore) {
    std::shared_ptr<Job> job = std::make_shared<Job>();

    bool succ = ::mmcInstantiatePendingJob(hCore, job->hJob);
    if (!succ) return false;

    jobs.push_back(job);
    return true;

    //// Remove this job from the list of instanc names. We only want
    //// views in there.
    //TCHAR instanceId[1024];
    //unsigned int len = sizeof(instanceId) / sizeof(TCHAR);
    //mmcGetInstanceID(jobHandle->operator void*(), instanceId, &len);
    //instanceNames.Remove(vislib::TString(instanceId));

}

megamol::console::JobManager::Job::Job() : hJob(), started(false) {
    // intentionally empty
}

megamol::console::JobManager::Job::~Job() {
    // intentionally empty
}

bool megamol::console::JobManager::Job::IsRunning() {
    return ::mmcIsJobRunning(hJob);
}

void megamol::console::JobManager::Job::Start() {
    if (!started) {
        started = ::mmcStartJob(hJob);
    }    
}
