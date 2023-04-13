/*
 * DataWriterJob.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "mmstd/data/DataWriterJob.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "mmcore/utility/log/Log.h"
#include "mmstd/data/DataWriterCtrlCall.h"

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
bool job::DataWriterJob::create() {
    return true; // intentionally empty ATM
}


/*
 * job::DataWriterJob::release
 */
void job::DataWriterJob::release() {
    // intentionally empty ATM
}


/*
 * job::DataWriterJob::Terminate
 */
bool job::DataWriterJob::Terminate() {
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
        Log::DefaultLog.WriteWarn("Data writer job not connected to any data writer module\n");
        return -1;
    }

    if (!(*dwcc)(DataWriterCtrlCall::CALL_GETCAPABILITIES)) {
        Log::DefaultLog.WriteWarn("Unable to query data writer capabilities");
    } else {
        this->abortable = dwcc->IsAbortable();
    }

    Log::DefaultLog.WriteInfo("Starting DataWriterJob \"%s\"", this->FullName().PeekBuffer());

    if ((*dwcc)(DataWriterCtrlCall::CALL_RUN)) {
        Log::DefaultLog.WriteInfo("DataWriterJob \"%s\" complete", this->FullName().PeekBuffer());

    } else {
        Log::DefaultLog.WriteWarn("DataWriterJob \"%s\" terminated with false", this->FullName().PeekBuffer());
        return -2;
    }

    return 0;
}
