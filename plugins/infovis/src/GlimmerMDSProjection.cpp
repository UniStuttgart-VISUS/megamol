#include "stdafx.h"
#include "GlimmerMDSProjection.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmstd_datatools/floattable/CallFloatTableData.h"

#include <sstream>
#include "glimmer.h"

using namespace megamol;
using namespace megamol::infovis;

GlimmerMDSProjection::GlimmerMDSProjection(void)
    : megamol::core::Module()
    , dataOutSlot("dataOut", "Ouput")
    , dataInSlot("dataIn", "Input")
    , reduceToNSlot("nEmbedding", "Number of dimensions to keep")
    , datahash(0)
    , dataInHash(0)
    , columnInfos() {
    this->dataInSlot.SetCompatibleCall<megamol::stdplugin::datatools::floattable::CallFloatTableDataDescription>();
    this->MakeSlotAvailable(&this->dataInSlot);

    this->dataOutSlot.SetCallback(megamol::stdplugin::datatools::floattable::CallFloatTableData::ClassName(),
        megamol::stdplugin::datatools::floattable::CallFloatTableData::FunctionName(0),
        &GlimmerMDSProjection::getDataCallback);
    this->dataOutSlot.SetCallback(megamol::stdplugin::datatools::floattable::CallFloatTableData::ClassName(),
        megamol::stdplugin::datatools::floattable::CallFloatTableData::FunctionName(1),
        &GlimmerMDSProjection::getHashCallback);
    this->MakeSlotAvailable(&this->dataOutSlot);

    reduceToNSlot << new ::megamol::core::param::IntParam(2);
    this->MakeSlotAvailable(&reduceToNSlot);
}

GlimmerMDSProjection::~GlimmerMDSProjection(void) { this->Release(); }

bool GlimmerMDSProjection::create(void) { return true; }

void GlimmerMDSProjection::release(void) {}

bool GlimmerMDSProjection::getDataCallback(core::Call& c) {
    try {
        megamol::stdplugin::datatools::floattable::CallFloatTableData* outCall =
            dynamic_cast<megamol::stdplugin::datatools::floattable::CallFloatTableData*>(&c);
        if (outCall == NULL) return false;

        megamol::stdplugin::datatools::floattable::CallFloatTableData* inCall =
            this->dataInSlot.CallAs<megamol::stdplugin::datatools::floattable::CallFloatTableData>();
        if (inCall == NULL) return false;

        inCall->SetFrameID(outCall->GetFrameID());
        if (!(*inCall)()) return false;

        bool finished = computeProjection(inCall);
        if (finished == false) return false;

        outCall->SetFrameCount(inCall->GetFrameCount());
        outCall->SetDataHash(this->datahash);

        // set outCall
        if (this->columnInfos.size() != 0) {
            outCall->Set(this->columnInfos.size(), this->data.size() / this->columnInfos.size(),
                this->columnInfos.data(), this->data.data());
        } else {
            outCall->Set(0, 0, NULL, NULL);
        }

    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError(_T("Failed to execute %hs::processData\n"), ClassName());
        return false;
    }

    return true;
}

bool GlimmerMDSProjection::getHashCallback(core::Call& c) {
    try {
        megamol::stdplugin::datatools::floattable::CallFloatTableData* outCall =
            dynamic_cast<megamol::stdplugin::datatools::floattable::CallFloatTableData*>(&c);
        if (outCall == NULL) return false;

        megamol::stdplugin::datatools::floattable::CallFloatTableData* inCall =
            this->dataInSlot.CallAs<megamol::stdplugin::datatools::floattable::CallFloatTableData>();
        if (inCall == NULL) return false;

        inCall->SetFrameID(outCall->GetFrameID());
        if (!(*inCall)(1)) return false;

        outCall->SetFrameCount(inCall->GetFrameCount());
        outCall->SetDataHash(this->datahash);
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError(_T("Failed to execute %hs::getHashCallback\n"), ClassName());
        return false;
    }

    return true;
}

bool megamol::infovis::GlimmerMDSProjection::computeProjection(
    megamol::stdplugin::datatools::floattable::CallFloatTableData* inCall) {
    // check if inData has changed and if Slots have changed
    if (this->dataInHash == inCall->DataHash()) {
        if (!reduceToNSlot.IsDirty()) {
            return true; // Nothing to do
        }
    }

    auto columnCount = inCall->GetColumnsCount();
    auto column_infos = inCall->GetColumnsInfos();
    auto rowsCount = inCall->GetRowsCount();
    auto inData = inCall->GetData();

    unsigned int outputColumnCount = this->reduceToNSlot.Param<core::param::IntParam>()->Value();

    if (outputColumnCount <= 0 || outputColumnCount > columnCount) {
        vislib::sys::Log::DefaultLog.WriteError(_T("%hs: No valid Dimension Count has been given\n"), ClassName());
        return false;
    }

    // Copy data to row-major format.
    float* inputData = (float*)malloc(columnCount * rowsCount * sizeof(float));
    for (int row = 0; row < rowsCount; row++) {
        for (int col = 0; col < columnCount; col++) {
            inputData[row * columnCount + col] = inData[row * columnCount + col];
        }
    }

    // Run the algorithm.
    float* outputData = nullptr;
    glimmer_mds(&outputData, outputColumnCount, inputData, columnCount, rowsCount);

    // Extrema search.
    float* max_vals = new float[outputColumnCount];
    float* min_vals = new float[outputColumnCount];
    for (int i = 0; i < outputColumnCount; i++) {
        max_vals[i] = std::numeric_limits<float>::min();
        min_vals[i] = std::numeric_limits<float>::max();
    }
    for (int row = 0; row < rowsCount; row++) {
        for (int col = 0; col < outputColumnCount; col++) {
            float value = outputData[row * outputColumnCount + col];
            if (value > max_vals[col]) {
                max_vals[col] = value;
            }
            if (value < min_vals[col]) {
                min_vals[col] = value;
            }
        }
    }

    // Generate columns.
    this->columnInfos.clear();
    this->columnInfos.resize(outputColumnCount);
    for (int col = 0; col < outputColumnCount; col++) {
        this->columnInfos[col]
            .SetName("Glimmer" + std::to_string(col))
            .SetType(megamol::stdplugin::datatools::floattable::CallFloatTableData::ColumnType::QUANTITATIVE)
            .SetMinimumValue(min_vals[col])
            .SetMaximumValue(max_vals[col]);
    }

    // Copy values.
    this->data.clear();
    this->data.reserve(rowsCount * outputColumnCount);
    for (size_t row = 0; row < rowsCount; row++) {
        for (size_t col = 0; col < outputColumnCount; col++)
            this->data.push_back(outputData[row * outputColumnCount + col]);
    }

    this->dataInHash = inCall->DataHash();
    this->datahash++;
    reduceToNSlot.ResetDirty();

    free(outputData);
    free(inputData);
    delete[] max_vals;
    delete[] min_vals;

    return true;
}
