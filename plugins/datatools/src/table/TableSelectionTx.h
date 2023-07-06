/**
 * MegaMol
 * Copyright (c) 2020, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <condition_variable>
#include <thread>

#include <zmq.hpp>

#include "datatools/table/TableDataCall.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

namespace megamol::datatools::table {

/*
 * Module to send table selection to another process.
 */
class TableSelectionTx : public core::Module {
public:
    /** Return module class name */
    static const char* ClassName() {
        return "TableSelectionTx";
    }

    /** Return module class description */
    static const char* Description() {
        return "Sends table selection to an other process.";
    }

    /** Module is always available */
    static bool IsAvailable() {
        return true;
    }
    /** Ctor */
    TableSelectionTx();

    /** Dtor */
    ~TableSelectionTx() override;

protected:
    bool create() override;

    void release() override;

    bool readDataCallback(core::Call& call);

    bool readMetaDataCallback(core::Call& call);

    bool writeDataCallback(core::Call& call);

    bool writeMetaDataCallback(core::Call& call);

    bool validateCalls();

    bool validateSelectionUpdate();

    void selectionSender();

    void selectionReceiver();

private:
    core::CallerSlot tableInSlot;
    core::CallerSlot flagStorageReadInSlot;
    core::CallerSlot flagStorageWriteInSlot;

    core::CalleeSlot flagStorageReadOutSlot;
    core::CalleeSlot flagStorageWriteOutSlot;

    core::param::ParamSlot updateSelectionParam;
    core::param::ParamSlot useColumnAsIndexParam;
    core::param::ParamSlot indexColumnParam;

    std::thread senderThread_;
    std::thread receiverThread_;
    std::unique_ptr<zmq::context_t> context_;
    std::vector<uint64_t> selected_;
    std::mutex selectedMutex_;
    std::condition_variable condVar_;
    bool senderThreadQuit_;
    bool senderThreadNotified_;
    bool receiverThreadQuit_;

    std::vector<uint64_t> receivedSelection_;
    bool receivedSelectionUpdate_;
    std::mutex receivedSelectionMutex_;
};

} // namespace megamol::datatools::table
