/*
 * AbstractJob.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "mmcore/job/AbstractJob.h"
#include "mmcore/AbstractNamedObject.h"
#include "mmcore/utility/log/Log.h"


using namespace megamol::core;


/*
 * job::AbstractJob::AbstractJob
 */
job::AbstractJob::AbstractJob(void) {
    // intentionally empty
}


/*
 * job::AbstractJob::~AbstractJob
 */
job::AbstractJob::~AbstractJob(void) {
    // intentionally empty
}


/*
 * job::AbstractJob::signalStart
 */
void job::AbstractJob::signalStart(void) {
    vislib::StringA name("unknown");
    AbstractNamedObject* ano = dynamic_cast<AbstractNamedObject*>(this);
    if (ano != NULL) {
        name = ano->Name();
    }

    megamol::core::utility::log::Log::DefaultLog.WriteInfo("Job \"%s\" started ...\n", name.PeekBuffer());
}


/*
 * job::AbstractJob::signalEnd
 */
void job::AbstractJob::signalEnd(bool wasTerminated) {
    vislib::StringA name("unknown");
    AbstractNamedObject* ano = dynamic_cast<AbstractNamedObject*>(this);
    if (ano != NULL) {
        name = ano->Name();
    }

    megamol::core::utility::log::Log::DefaultLog.WriteInfo(
        "Job \"%s\" %s\n", name.PeekBuffer(), wasTerminated ? "terminated" : "finished");

    // Informing the core about job termination is not required because the
    // frontend main loop pools for running jobs and will close the handle
    // when the job is finished.
}
