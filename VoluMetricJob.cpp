/*
 * VoluMetricJob.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "VoluMetricJob.h"
#include "moldyn/MultiParticleDataCall.h"
#include "param/FilePathParam.h"
#include "vislib/Log.h"

using namespace megamol;
using namespace megamol::trisoup;


/*
 * VoluMetricJob::VoluMetricJob
 */
VoluMetricJob::VoluMetricJob(void) : core::job::AbstractThreadedJob(), core::Module(),
        getDataSlot("getData", "TODO: Description here!"),
        resultFilenameSlot("resultFilename", "TODO: Description here!") {

    this->getDataSlot.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

    this->resultFilenameSlot << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->resultFilenameSlot);

    // TODO: Implement

}


/*
 * VoluMetricJob::~VoluMetricJob
 */
VoluMetricJob::~VoluMetricJob(void) {
    this->Release();
}


/*
 * VoluMetricJob::create
 */
bool VoluMetricJob::create(void) {

    // TODO: Implement

    return true;
}


/*
 * VoluMetricJob::release
 */
void VoluMetricJob::release(void) {

    // TODO: Implement

}


/*
 * VoluMetricJob::Run
 */
DWORD VoluMetricJob::Run(void *userData) {
    using vislib::sys::Log;

    core::moldyn::MultiParticleDataCall *datacall = this->getDataSlot.CallAs<core::moldyn::MultiParticleDataCall>();
    if (datacall == NULL) {
        Log::DefaultLog.WriteError("No data source connected to VoluMetricJob");
        return -1;
    }

    if (!(*datacall)(1)) {
        Log::DefaultLog.WriteError("Data source does not answer to extent request");
        return -2;
    }

    unsigned int frameCnt = datacall->FrameCount();
    Log::DefaultLog.WriteInfo("Data source with %u frame(s)", frameCnt);

    for (unsigned int frameI = 0; frameI < frameCnt; frameI++) {

        datacall->SetFrameID(frameI, true);
        if (!(*datacall)(0)) {
            Log::DefaultLog.WriteError("ARG! No frame here", frameI);
            return -3;
        }
        while (datacall->FrameID() != frameI) {
            vislib::sys::Thread::Sleep(100);
            datacall->SetFrameID(frameI, true);
            if (!(*datacall)(0)) {
                Log::DefaultLog.WriteError("ARG! No frame here", frameI);
                return -3;
            }
        }

        unsigned int partListCnt = datacall->GetParticleListCount();
        for (unsigned int partListI = 0; partListI < partListCnt; partListI++) {
            printf("%u particle in list %u\n", datacall->AccessParticles(partListI).GetCount(), partListI);
        }
        printf("Fuer Teschzwecke\n");

        // TODO: Implement

    }

    return 0;
}
