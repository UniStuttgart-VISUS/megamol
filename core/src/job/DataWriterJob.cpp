/*
 * DataWriterJob.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "mmcore/job/DataWriterJob.h"
#include "mmcore/DataWriterCtrlCall.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "mmcore/utility/log/Log.h"

using namespace megamol::core;


/*
 * job::DataWriterJob::DataWriterJob
 */
job::DataWriterJob::DataWriterJob()
        : AbstractThreadedJob()
        , Module()
        , writerSlot("writer", "Slot to the controlled writer module")
        , abortable(false) {
    this->writerSlot.SetCompatibleCall<factories::CallAutoDescription<DataWriterCtrlCall>>();
    this->MakeSlotAvailable(&this->writerSlot);
}


/*
 * job::DataWriterJob::~DataWriterJob
 */
job::DataWriterJob::~DataWriterJob() {
    this->Release();
}


/*
 * job::DataWriterJob::create
 */
bool job::DataWriterJob::create(void) {
    return true; // intentionally empty ATM
}


/*
 * job::DataWriterJob::release
 */
void job::DataWriterJob::release(void) {
    // intentionally empty ATM
}


/*
 * job::DataWriterJob::Terminate
 */
bool job::DataWriterJob::Terminate(void) {
    AbstractThreadedJob::Terminate();
    if (this->abortable) {
        DataWriterCtrlCall* dwcc = this->writerSlot.CallAs<DataWriterCtrlCall>();
        if (dwcc != NULL) {
            return (*dwcc)(DataWriterCtrlCall::CALL_ABORT);
        }
    }
    return false;
}


/*
 * job::DataWriterJob::Run
 */
DWORD job::DataWriterJob::Run(void* userData) {
    using megamol::core::utility::log::Log;
    DataWriterCtrlCall* dwcc = this->writerSlot.CallAs<DataWriterCtrlCall>();

    if (dwcc == NULL) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_WARN, "Data writer job not connected to any data writer module\n");
        return -1;
    }

    if (!(*dwcc)(DataWriterCtrlCall::CALL_GETCAPABILITIES)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_WARN, "Unable to query data writer capabilities");
    } else {
        this->abortable = dwcc->IsAbortable();
    }

    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Starting DataWriterJob \"%s\"", this->FullName().PeekBuffer());

    if ((*dwcc)(DataWriterCtrlCall::CALL_RUN)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "DataWriterJob \"%s\" complete", this->FullName().PeekBuffer());

    } else {
        Log::DefaultLog.WriteMsg(
            Log::LEVEL_WARN, "DataWriterJob \"%s\" terminated with false", this->FullName().PeekBuffer());
        return -2;
    }

    return 0;
}
