/*
 * TableProcessorBase.cpp
 *
 * Copyright (C) 2019 Visualisierungsinstitut der Universität Stuttgart
 * Alle Rechte vorbehalten.
 */

#include "TableProcessorBase.h"

#include <cassert>
#include <limits>

#include "mmcore/utility/log/Log.h"

/*
 * megamol::datatools::table::TableProcessorBase::TableProcessorBase
 */
megamol::datatools::table::TableProcessorBase::TableProcessorBase()
        : frameID((std::numeric_limits<unsigned int>::max)())
        , inputHash(0)
        , localHash(0)
        , slotInput("input", "The input slot providing the unfiltered data.")
        , slotOutput("output", "The input slot for the filtered data.") {
    /* Export the calls. */
    this->slotInput.SetCompatibleCall<TableDataCallDescription>();
    this->MakeSlotAvailable(&this->slotInput);

    this->slotOutput.SetCallback(
        TableDataCall::ClassName(), TableDataCall::FunctionName(0), &TableProcessorBase::getData);
    this->slotOutput.SetCallback(
        TableDataCall::ClassName(), TableDataCall::FunctionName(1), &TableProcessorBase::getHash);
    this->MakeSlotAvailable(&this->slotOutput);
}


/*
 * megamol::datatools::table::TableProcessorBase::getData
 */
bool megamol::datatools::table::TableProcessorBase::getData(core::Call& call) {
    using namespace core::param;
    using megamol::core::utility::log::Log;

    auto src = this->slotInput.CallAs<TableDataCall>();
    auto dst = dynamic_cast<TableDataCall*>(&call);

    /* Sanity checks. */
    if (src == nullptr) {
        Log::DefaultLog.WriteError(_T("The input slot of %hs is invalid"), TableDataCall::ClassName());
        return false;
    }

    if (dst == nullptr) {
        Log::DefaultLog.WriteError(_T("The output slot of %hs is invalid"), TableDataCall::ClassName());
        return false;
    }

    if (!this->prepareData(*src, dst->GetFrameID())) {
        return false;
    }

    dst->SetFrameCount(src->GetFrameCount());
    dst->SetFrameID(this->frameID);
    dst->SetDataHash(this->getHash());
    dst->Set(
        this->columns.size(), this->values.size() / this->columns.size(), this->columns.data(), this->values.data());

    return true;
}


/*
 * megamol::datatools::table::TableProcessorBase::getHash
 */
bool megamol::datatools::table::TableProcessorBase::getHash(core::Call& call) {
    using megamol::core::utility::log::Log;
    auto src = this->slotInput.CallAs<TableDataCall>();
    auto dst = dynamic_cast<TableDataCall*>(&call);

    /* Sanity checks. */
    if (src == nullptr) {
        Log::DefaultLog.WriteError("The input slot of type %hs is invalid", TableDataCall::ClassName());
        return false;
    }

    if (dst == nullptr) {
        Log::DefaultLog.WriteError("The output slot of type %hs is invalid", TableDataCall::ClassName());
        return false;
    }

    /* Obtain extents and hash of the source data. */
    src->SetFrameID(dst->GetFrameID());
    if (!(*src)(1)) {
        Log::DefaultLog.WriteError(
            "The call to %hs of %hs failed.", TableDataCall::FunctionName(1), TableDataCall::ClassName());
        return false;
    }

    // I don't know why the getHash call passes on the frame count, but it seems
    // to be the expected behaviour ...
    {
        auto cnt = src->GetFrameCount();
        dst->SetFrameCount(cnt);
    }

    dst->SetDataHash(this->getHash());
    dst->SetUnlocker(nullptr);

    return true;
}
