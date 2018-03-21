#include "stdafx.h"
#include "LinesToFloatTable.h"

#include "geometry_calls/LinesDataCall.h"

megamol::stdplugin::datatools::LinesToFloatTable::LinesToFloatTable(void)
    : megamol::core::Module()
    , dataOutSlot("dataOut", "data output")
    , dataInSlot("dataIn", "data input")
    , frameID{std::numeric_limits<unsigned int>::max()}
    , datahash{std::numeric_limits<size_t>::max()} {
    this->dataOutSlot.SetCallback(megamol::stdplugin::datatools::floattable::CallFloatTableData::ClassName(),
        megamol::stdplugin::datatools::floattable::CallFloatTableData::FunctionName(0), &LinesToFloatTable::getDataCallback);
    this->dataOutSlot.SetCallback(megamol::stdplugin::datatools::floattable::CallFloatTableData::ClassName(),
        megamol::stdplugin::datatools::floattable::CallFloatTableData::FunctionName(1), &LinesToFloatTable::getDataCallback);
    this->MakeSlotAvailable(&this->dataOutSlot);

    this->dataInSlot.SetCompatibleCall<megamol::geocalls::LinesDataCallDescription>();
    this->MakeSlotAvailable(&this->dataInSlot);
}


megamol::stdplugin::datatools::LinesToFloatTable::~LinesToFloatTable(void) { this->Release(); }


bool megamol::stdplugin::datatools::LinesToFloatTable::create(void) {
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


void megamol::stdplugin::datatools::LinesToFloatTable::release(void) {}


bool megamol::stdplugin::datatools::LinesToFloatTable::getDataCallback(megamol::core::Call& c) {
    megamol::stdplugin::datatools::floattable::CallFloatTableData* outCall =
        dynamic_cast<megamol::stdplugin::datatools::floattable::CallFloatTableData*>(&c);
    if (outCall == nullptr) return false;

    megamol::geocalls::LinesDataCall* inCall = this->dataInSlot.CallAs<megamol::geocalls::LinesDataCall>();
    if (inCall == nullptr) return false;

    if (!this->assertData(outCall)) return false;

    return true;
}


bool megamol::stdplugin::datatools::LinesToFloatTable::assertData(
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

        auto const lineCount = inCall->Count();

        //float edgeId = 0.0f;

        this->data.clear();

        for (unsigned int li = 0; li < lineCount; ++li) {
            auto const& line = inData[li];
            auto const numElements = line.Count();
            auto const indexArrayDT = line.IndexArrayDataType();
            if (indexArrayDT == geocalls::LinesDataCall::Lines::DT_NONE) {
                // numElements represents number of vertices
                for (unsigned int vi = 0; vi < numElements; ++vi) {
                    this->data.push_back(li);
                    auto vertex = line[vi];
                    this->data.push_back(vertex.vert.GetXf());
                    this->data.push_back(vertex.vert.GetYf());
                    this->data.push_back(vertex.vert.GetZf());
                    /*this->data.push_back(edgeId);
                    vertex = line[vi + 1];
                    this->data.push_back(vertex.vert.GetXf());
                    this->data.push_back(vertex.vert.GetYf());
                    this->data.push_back(vertex.vert.GetZf());
                    edgeId += 1.0f;*/
                }
            } else {
                // numElements represents number of indices
                for (unsigned int i = 0; i < numElements; ++i) {
                    this->data.push_back(li);
                    auto idx = line.GetIdx(i * 2);
                    auto vertex = line[idx.idx.GetIDXu32()];
                    this->data.push_back(vertex.vert.GetXf());
                    this->data.push_back(vertex.vert.GetYf());
                    this->data.push_back(vertex.vert.GetZf());
                    /*this->data.push_back(edgeId);
                    idx = line.GetIdx(i * 2 + 1);
                    vertex = line[idx.idx.GetIDXu32()];
                    this->data.push_back(vertex.vert.GetXf());
                    this->data.push_back(vertex.vert.GetYf());
                    this->data.push_back(vertex.vert.GetZf());
                    edgeId += 1.0f;*/
                }
            }
        }
    }

    outCall->SetDataHash(this->datahash);
    outCall->SetFrameCount(inCall->FrameCount());
    outCall->SetFrameID(this->frameID);
    outCall->Set(this->columnInfos.size(), this->data.size() / this->columnInfos.size(), this->columnInfos.data(), this->data.data());

    return true;
}
