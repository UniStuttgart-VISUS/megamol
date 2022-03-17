/*
 * TableSelectionTx.cpp
 *
 * Copyright (C) 2020 by VISUS (University of Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "TableSelectionTx.h"

#include <unordered_set>

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/utility/log/Log.h"
#include "mmcore_gl/flags/FlagCallsGL.h"

using namespace megamol::datatools_gl;
using namespace megamol::datatools_gl::table;
using namespace megamol;

TableSelectionTx::TableSelectionTx()
        : core::Module()
        , tableInSlot("getTableIn", "Float table input")
        , flagStorageReadInSlot("readFlagStorageIn", "Flag storage read input")
        , flagStorageWriteInSlot("writeFlagStorageIn", "Flag storage write input")
        , flagStorageReadOutSlot("readFlagStorageOut", "Flag storage read output")
        , flagStorageWriteOutSlot("writeFlagStorageOut", "Flag storage write output")
        , updateSelectionParam("updateSelection", "Enable selection update")
        , useColumnAsIndexParam("useColumnAsIndex", "Use column as index instead of row id")
        , indexColumnParam("indexColumn", "Numeric index of column, which is used as row index")
        , senderThreadQuit_(false)
        , senderThreadNotified_(false)
        , receiverThreadQuit_(false)
        , receivedSelectionUpdate_(false) {
    this->tableInSlot.SetCompatibleCall<datatools::table::TableDataCallDescription>();
    this->MakeSlotAvailable(&this->tableInSlot);

    this->flagStorageReadInSlot.SetCompatibleCall<core_gl::FlagCallRead_GLDescription>();
    this->MakeSlotAvailable(&this->flagStorageReadInSlot);

    this->flagStorageWriteInSlot.SetCompatibleCall<core_gl::FlagCallWrite_GLDescription>();
    this->MakeSlotAvailable(&this->flagStorageWriteInSlot);

    this->flagStorageReadOutSlot.SetCallback(core_gl::FlagCallRead_GL::ClassName(),
        core_gl::FlagCallRead_GL::FunctionName(core_gl::FlagCallRead_GL::CallGetData),
        &TableSelectionTx::readDataCallback);
    this->flagStorageReadOutSlot.SetCallback(core_gl::FlagCallRead_GL::ClassName(),
        core_gl::FlagCallRead_GL::FunctionName(core_gl::FlagCallRead_GL::CallGetMetaData),
        &TableSelectionTx::readMetaDataCallback);
    this->MakeSlotAvailable(&this->flagStorageReadOutSlot);

    this->flagStorageWriteOutSlot.SetCallback(core_gl::FlagCallWrite_GL::ClassName(),
        core_gl::FlagCallWrite_GL::FunctionName(core_gl::FlagCallWrite_GL::CallGetData),
        &TableSelectionTx::writeDataCallback);
    this->flagStorageWriteOutSlot.SetCallback(core_gl::FlagCallWrite_GL::ClassName(),
        core_gl::FlagCallWrite_GL::FunctionName(core_gl::FlagCallWrite_GL::CallGetMetaData),
        &TableSelectionTx::writeMetaDataCallback);
    this->MakeSlotAvailable(&this->flagStorageWriteOutSlot);

    this->updateSelectionParam << new core::param::BoolParam(true);
    this->MakeSlotAvailable(&this->updateSelectionParam);

    this->useColumnAsIndexParam << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->useColumnAsIndexParam);

    this->indexColumnParam << new core::param::IntParam(0, 0);
    this->MakeSlotAvailable(&this->indexColumnParam);
}

TableSelectionTx::~TableSelectionTx() {
    this->Release();
}

bool TableSelectionTx::create() {
    context_ = std::make_unique<zmq::context_t>(1);

    senderThreadQuit_ = false;
    if (!senderThread_.joinable()) {
        senderThread_ = std::thread(&TableSelectionTx::selectionSender, this);
    }

    receiverThreadQuit_ = false;
    if (!receiverThread_.joinable()) {
        receiverThread_ = std::thread(&TableSelectionTx::selectionReceiver, this);
    }

    return true;
}

void TableSelectionTx::release() {
    senderThreadQuit_ = true;
    receiverThreadQuit_ = true;
    senderThreadNotified_ = true;
    condVar_.notify_one();
    context_->close();
    context_.reset();
    if (senderThread_.joinable()) {
        senderThread_.join();
    }
    if (receiverThread_.joinable()) {
        receiverThread_.join();
    }
}

bool TableSelectionTx::readDataCallback(core::Call& call) {
    auto* flagsReadOutCall = dynamic_cast<core_gl::FlagCallRead_GL*>(&call);
    if (flagsReadOutCall == nullptr) {
        return false;
    }

    if (!validateCalls()) {
        return false;
    }

    if (!validateSelectionUpdate()) {
        return false;
    }

    auto* flagsReadInCall = this->flagStorageReadInSlot.CallAs<core_gl::FlagCallRead_GL>();

    (*flagsReadInCall)(core_gl::FlagCallRead_GL::CallGetData);
    flagsReadOutCall->setData(flagsReadInCall->getData(), flagsReadInCall->version());

    return true;
}

bool TableSelectionTx::readMetaDataCallback(core::Call& call) {
    // FlagCall_GL has empty meta data
    return true;
}

bool TableSelectionTx::writeDataCallback(core::Call& call) {
    auto* flagsWriteOutCall = dynamic_cast<core_gl::FlagCallWrite_GL*>(&call);
    if (flagsWriteOutCall == nullptr) {
        return false;
    }

    if (!validateCalls()) {
        return false;
    }

    auto* flagsWriteInCall = this->flagStorageWriteInSlot.CallAs<core_gl::FlagCallWrite_GL>();

    flagsWriteInCall->setData(flagsWriteOutCall->getData(), flagsWriteOutCall->version());
    (*flagsWriteInCall)(core_gl::FlagCallWrite_GL::CallGetData);

    // Send data

    auto* tableInCall = this->tableInSlot.CallAs<datatools::table::TableDataCall>();

    tableInCall->SetFrameID(0);
    (*tableInCall)(1);
    (*tableInCall)(0);

    bool useColumnAsIndex = this->useColumnAsIndexParam.Param<core::param::BoolParam>()->Value();
    int indexColumn = this->indexColumnParam.Param<core::param::IntParam>()->Value();

    auto flags = flagsWriteOutCall->getData()->flags;
    size_t numberOfFlags = flags->getByteSize() / sizeof(uint32_t);
    size_t numberOfRows = tableInCall->GetRowsCount();
    size_t numberOfCols = tableInCall->GetColumnsCount();
    auto* tableData = tableInCall->GetData();

    if (indexColumn >= numberOfCols) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("TableSelectionTx: invalid column index!");
        return false;
    }

    // validateFlagCount() only increases the buffer, therefore numberOfFlags > numberOfRows is also valid.
    if (numberOfFlags < numberOfRows) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("TableSelectionTx: invalid table/flag storage size!");
        return false;
    }

    std::vector<uint32_t> flagsData(numberOfFlags);
    flags->bind();
    glGetBufferSubData(flags->getTarget(), 0, flags->getByteSize(), flagsData.data());

    core::FlagStorageTypes::flag_item_type testMask = core::FlagStorageTypes::to_integral(
        core::FlagStorageTypes::flag_bits::ENABLED | core::FlagStorageTypes::flag_bits::FILTERED);
    constexpr core::FlagStorageTypes::flag_item_type passMask =
        core::FlagStorageTypes::to_integral(core::FlagStorageTypes::flag_bits::ENABLED);

    std::unique_lock<std::mutex> lock(selectedMutex_);
    selected_.clear();
    for (size_t i = 0; i < numberOfRows; ++i) {
        if ((flagsData[i] & testMask) == passMask) {
            if (flagsData[i] & core::FlagStorageTypes::to_integral(core::FlagStorageTypes::flag_bits::SELECTED)) {
                if (useColumnAsIndex) {
                    selected_.push_back(static_cast<uint64_t>(tableData[indexColumn + i * numberOfCols]));
                } else {
                    selected_.push_back(static_cast<uint64_t>(i));
                }
            } else {
                // not selected
            }
        }
    }
    senderThreadNotified_ = true;
    condVar_.notify_one();
    lock.unlock();

    return true;
}

bool TableSelectionTx::writeMetaDataCallback(core::Call& call) {
    // FlagCall_GL has empty meta data
    return true;
}

bool TableSelectionTx::validateCalls() {
    auto* tableInCall = this->tableInSlot.CallAs<datatools::table::TableDataCall>();
    auto* flagsReadInCall = this->flagStorageReadInSlot.CallAs<core_gl::FlagCallRead_GL>();
    auto* flagsWriteInCall = this->flagStorageWriteInSlot.CallAs<core_gl::FlagCallWrite_GL>();

    if (tableInCall == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("TableSelectionTx requires a table!");
        return false;
    }

    if (flagsReadInCall == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("TableSelectionTx requires a read flag storage!");
        return false;
    }

    if (flagsWriteInCall == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("TableSelectionTx requires a write flag storage!");
        return false;
    }

    return true;
}

bool TableSelectionTx::validateSelectionUpdate() {
    std::lock_guard<std::mutex> lock(receivedSelectionMutex_);
    if (!receivedSelectionUpdate_) {
        return true;
    }

    receivedSelectionUpdate_ = false;

    if (!this->updateSelectionParam.Param<core::param::BoolParam>()->Value()) {
        return true;
    }

    auto* flagsReadInCall = this->flagStorageReadInSlot.CallAs<core_gl::FlagCallRead_GL>();
    (*flagsReadInCall)(core_gl::FlagCallRead_GL::CallGetData);
    auto flagCollection = flagsReadInCall->getData();
    auto version = flagsReadInCall->version();

    auto* tableInCall = this->tableInSlot.CallAs<datatools::table::TableDataCall>();
    tableInCall->SetFrameID(0);
    (*tableInCall)(1);
    (*tableInCall)(0);

    size_t numberOfRows = tableInCall->GetRowsCount();

    core::FlagStorageTypes::flag_vector_type flags_data(
        numberOfRows, core::FlagStorageTypes::to_integral(core::FlagStorageTypes::flag_bits::ENABLED));

    if (!this->useColumnAsIndexParam.Param<core::param::BoolParam>()->Value()) {
        // Select received rows
        for (auto id : receivedSelection_) {
            if (id >= 0 && id < numberOfRows) {
                flags_data[id] |= core::FlagStorageTypes::to_integral(core::FlagStorageTypes::flag_bits::SELECTED);
            }
        }
    } else {
        // Select received values
        size_t numberOfCols = tableInCall->GetColumnsCount();
        auto* tableData = tableInCall->GetData();
        int indexColumn = this->indexColumnParam.Param<core::param::IntParam>()->Value();
        if (indexColumn >= numberOfCols) {
            megamol::core::utility::log::Log::DefaultLog.WriteError("TableSelectionTx: invalid column index!");
            return false;
        }

        // Use unordered set for O(1) lookup of selected values
        std::unordered_set<float> s;
        for (auto id : receivedSelection_) {
            s.insert(static_cast<float>(id));
        }

        for (size_t i = 0; i < numberOfRows; i++) {
            float value = tableData[indexColumn + i * numberOfCols];
            if (s.find(value) != s.end()) {
                flags_data[i] |= core::FlagStorageTypes::to_integral(core::FlagStorageTypes::flag_bits::SELECTED);
            }
        }
    }

    flagCollection->flags =
        std::make_shared<glowl::BufferObject>(GL_SHADER_STORAGE_BUFFER, flags_data, GL_DYNAMIC_DRAW);

    auto* flagsWriteInCall = this->flagStorageWriteInSlot.CallAs<core_gl::FlagCallWrite_GL>();
    flagsWriteInCall->setData(flagCollection, version + 1);
    (*flagsWriteInCall)(core_gl::FlagCallWrite_GL::CallGetData);

    return true;
}

void TableSelectionTx::selectionSender() {
    zmq::socket_t socket{*context_, ZMQ_REQ};
    socket.setsockopt(ZMQ_LINGER, 0);
    socket.connect("tcp://localhost:10001");

    std::unique_lock<std::mutex> lock(selectedMutex_);
    while (!senderThreadQuit_) {
        while (!senderThreadNotified_ && !senderThreadQuit_) {
            condVar_.wait(lock);
        }
        while (senderThreadNotified_ && !senderThreadQuit_) {
            senderThreadNotified_ = false;
            try {
                zmq::message_t request{selected_.cbegin(), selected_.cend()};
                lock.unlock();
                socket.send(request, zmq::send_flags::none);

                zmq::message_t reply{};
                socket.recv(reply, zmq::recv_flags::none);
            } catch (const zmq::error_t& e) {
                if (e.num() != ETERM) {
                    std::cerr << e.what() << std::endl;
                }
            }
            lock.lock();
        }
    }
}

void TableSelectionTx::selectionReceiver() {
    zmq::socket_t socket{*context_, ZMQ_REP};
    socket.bind("tcp://*:10002");

    const std::string okString{"Ok!"};

    while (!receiverThreadQuit_) {
        try {
            zmq::message_t request;
            socket.recv(request, zmq::recv_flags::none);
            size_t size = request.size() / sizeof(uint64_t);

            if (size > 0) {
                std::lock_guard<std::mutex> lock(receivedSelectionMutex_);
                uint64_t* data_ptr = static_cast<uint64_t*>(request.data());
                receivedSelection_ = std::vector<uint64_t>(data_ptr, data_ptr + size);
                receivedSelectionUpdate_ = true;
            }

            zmq::message_t reply{okString.cbegin(), okString.cend()};
            socket.send(reply, zmq::send_flags::none);
        } catch (const zmq::error_t& e) {
            if (e.num() != ETERM) {
                std::cerr << e.what() << std::endl;
            }
        }
    }
}
