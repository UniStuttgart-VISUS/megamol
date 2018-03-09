#include "stdafx.h"
#include "LinesToCSV.h"

#include "geometry_calls/LinesDataCall.h"

megamol::stdplugin::datatools::LinesToCSV::LinesToCSV(void)
    : megamol::core::Module()
    , dataOutSlot("dataOut", "data output")
    , dataInSlot("dataIn", "data input")
    , frameID{std::numeric_limits<unsigned int>::max()}
    , datahash{std::numeric_limits<size_t>::max()} {
    this->dataOutSlot.SetCallback(megamol::stdplugin::datatools::floattable::CallFloatTableData::ClassName(),
        megamol::stdplugin::datatools::floattable::CallFloatTableData::FunctionName(0), &LinesToCSV::getDataCallback);
    this->MakeSlotAvailable(&this->dataOutSlot);

    this->dataInSlot.SetCompatibleCall<megamol::geocalls::LinesDataCallDescription>();
    this->MakeSlotAvailable(&this->dataInSlot);
}


megamol::stdplugin::datatools::LinesToCSV::~LinesToCSV(void) { this->Release(); }


bool megamol::stdplugin::datatools::LinesToCSV::create(void) {
    megamol::stdplugin::datatools::floattable::CallFloatTableData::ColumnInfo ci;

    ci.SetName("edgeId");
    ci.SetType(megamol::stdplugin::datatools::floattable::CallFloatTableData::ColumnType::QUANTITATIVE);

    this->columnInfos.push_back(ci);

    ci.SetName("x");

    this->columnInfos.push_back(ci);

    ci.SetName("y");

    this->columnInfos.push_back(ci);

    ci.SetName("z");

    this->columnInfos.push_back(ci);

    return true;
}


void megamol::stdplugin::datatools::LinesToCSV::release(void) {}


bool megamol::stdplugin::datatools::LinesToCSV::getDataCallback(megamol::core::Call& c) {
    megamol::stdplugin::datatools::floattable::CallFloatTableData* outCall =
        dynamic_cast<megamol::stdplugin::datatools::floattable::CallFloatTableData*>(&c);
    if (outCall == nullptr) return false;

    megamol::geocalls::LinesDataCall* inCall = this->dataInSlot.CallAs<megamol::geocalls::LinesDataCall>();
    if (inCall == nullptr) return false;

    return true;
}


bool megamol::stdplugin::datatools::LinesToCSV::assertData(
    megamol::stdplugin::datatools::floattable::CallFloatTableData* outCall) {
    megamol::geocalls::LinesDataCall* inCall = this->dataInSlot.CallAs<megamol::geocalls::LinesDataCall>();
    if (inCall == nullptr) return false;

    auto const req_frameID = outCall->GetFrameID();

    if (req_frameID != this->frameID || this->datahash != inCall->DataHash()) {
        do {
            inCall->SetFrameID(req_frameID);
            if (!(*inCall)(1)) return false;
            if (!(*inCall)(0)) return false;
        } while (inCall->FrameID() != req_frameID);

        this->frameID = inCall->FrameID();
        this->datahash = inCall->DataHash();

        auto const inData = inCall->GetLines();

        if (inData == nullptr) return false;

        auto lineCount = inCall->Count();

        for (unsigned int li = 0; li < lineCount; ++li) {
            
        }
    }


    return true;
}
