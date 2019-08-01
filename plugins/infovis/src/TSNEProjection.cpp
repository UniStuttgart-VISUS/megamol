#include "stdafx.h"
#include "TSNEProjection.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmstd_datatools/table/TableDataCall.h"

#include <sstream>
#include <tsne.h>

using namespace megamol;
using namespace megamol::infovis;

TSNEProjection::TSNEProjection(void)
    : megamol::core::Module()
    , dataOutSlot("dataOut", "Ouput")
    , dataInSlot("dataIn", "Input")
    , reduceToNSlot("nComponents", "Number of components (dimensions) to keep")
    , randomSeedSlot("randomSeed", "Set the random Seed. Set to -1 for time dependant random Seed")
    , thetaSlot("gradientAccurancy",
          "theta = 0 corresponds to standard, slow t-SNE, while theta = 1 corresponds to very crude approximations")
    , maxIterSlot("maxIter", "Set the maximum Iterations")
    , perplexitySlot("perplexity", "Set the Perplexity")
    , datahash(0)
    , dataInHash(0)
    , columnInfos() {

    TSNE* tsne = new TSNE(); // lib load test

    this->dataInSlot.SetCompatibleCall<megamol::stdplugin::datatools::table::TableDataCallDescription>();
    this->MakeSlotAvailable(&this->dataInSlot);

    this->dataOutSlot.SetCallback(megamol::stdplugin::datatools::table::TableDataCall::ClassName(),
        megamol::stdplugin::datatools::table::TableDataCall::FunctionName(0), &TSNEProjection::getDataCallback);
    this->dataOutSlot.SetCallback(megamol::stdplugin::datatools::table::TableDataCall::ClassName(),
        megamol::stdplugin::datatools::table::TableDataCall::FunctionName(1), &TSNEProjection::getHashCallback);
    this->MakeSlotAvailable(&this->dataOutSlot);

    reduceToNSlot << new ::megamol::core::param::IntParam(2);
    this->MakeSlotAvailable(&reduceToNSlot);

    randomSeedSlot << new ::megamol::core::param::IntParam(42);
    this->MakeSlotAvailable(&randomSeedSlot);

    maxIterSlot << new ::megamol::core::param::IntParam(1000);
    this->MakeSlotAvailable(&maxIterSlot);

    perplexitySlot << new ::megamol::core::param::FloatParam(30);
    this->MakeSlotAvailable(&perplexitySlot);

    thetaSlot << new ::megamol::core::param::FloatParam(0.5);
    this->MakeSlotAvailable(&thetaSlot);
}

TSNEProjection::~TSNEProjection(void) { this->Release(); }

bool TSNEProjection::create(void) { return true; }

void TSNEProjection::release(void) {}

bool TSNEProjection::getDataCallback(core::Call& c) {
    try {
        megamol::stdplugin::datatools::table::TableDataCall* outCall =
            dynamic_cast<megamol::stdplugin::datatools::table::TableDataCall*>(&c);
        if (outCall == NULL) return false;

        megamol::stdplugin::datatools::table::TableDataCall* inCall =
            this->dataInSlot.CallAs<megamol::stdplugin::datatools::table::TableDataCall>();
        if (inCall == NULL) return false;

        inCall->SetFrameID(outCall->GetFrameID());
        if (!(*inCall)()) return false;

        bool finished = project(inCall);
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

bool TSNEProjection::getHashCallback(core::Call& c) {
    try {
        megamol::stdplugin::datatools::table::TableDataCall* outCall =
            dynamic_cast<megamol::stdplugin::datatools::table::TableDataCall*>(&c);
        if (outCall == NULL) return false;

        megamol::stdplugin::datatools::table::TableDataCall* inCall =
            this->dataInSlot.CallAs<megamol::stdplugin::datatools::table::TableDataCall>();
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

bool megamol::infovis::TSNEProjection::project(megamol::stdplugin::datatools::table::TableDataCall* inCall) {
    // check if inData has changed and if Slots have changed
    if (this->dataInHash == inCall->DataHash()) {
        if (!reduceToNSlot.IsDirty() && !maxIterSlot.IsDirty() && !thetaSlot.IsDirty() && !perplexitySlot.IsDirty() &&
            !randomSeedSlot.IsDirty()) {
            return true; // Nothing to do
        }
    }

    auto columnCount = inCall->GetColumnsCount();
    auto column_infos = inCall->GetColumnsInfos();
    auto rowsCount = inCall->GetRowsCount();
    auto inData = inCall->GetData();

    unsigned int outputColumnCount = this->reduceToNSlot.Param<core::param::IntParam>()->Value();
    int maxIter = this->maxIterSlot.Param<core::param::IntParam>()->Value();
    int randomSeed = this->randomSeedSlot.Param<core::param::IntParam>()->Value();
    double theta = this->thetaSlot.Param<core::param::FloatParam>()->Value();
    double perplexity = this->perplexitySlot.Param<core::param::FloatParam>()->Value();


    if (outputColumnCount <= 0 || outputColumnCount > columnCount) {
        vislib::sys::Log::DefaultLog.WriteError(_T("%hs: No valid Dimension Count has been given\n"), ClassName());
        return false;
    }

    // Load data in a double Array
    double* inputData = (double*)malloc(columnCount * rowsCount * sizeof(double));
    for (int col = 0; col < columnCount; col++) {
        for (int row = 0; row < rowsCount; row++) {
            inputData[row * columnCount + col] = inData[row * columnCount + col];
            // inputData[col*rowsCount + row] = inData[row * columnCount + col];
        }
    }


    TSNE* tsne = new TSNE();

    double* result = (double*)malloc(rowsCount * outputColumnCount * sizeof(double));
    // void run(double* X, int N, int D, double* Y, int no_dims, double perplexity, double theta, int rand_seed,
    // bool skip_random_init, int max_iter = 1000, int stop_lying_iter = 250, int mom_switch_iter = 250);
    tsne->run(
        inputData, rowsCount, columnCount, result, outputColumnCount, perplexity, theta, randomSeed, false, maxIter);

    double* maximas = new double[outputColumnCount];
    double* minimas = new double[outputColumnCount];

    for (int col = 0; col < outputColumnCount; col++) {
        maximas[col] = result[col];
        minimas[col] = result[col];
    }

    for (int col = 0; col < outputColumnCount; col++) {
        for (int row = 1; row < rowsCount; row++) {
            double value = result[row * outputColumnCount + col];
            if (maximas[col] < value) maximas[col] = value;
            if (minimas[col] > value) minimas[col] = value;
        }
    }


    // std::stringstream debug;
    // debug << std::endl << result << std::endl;
    // vislib::sys::Log::DefaultLog.WriteInfo(debug.str().c_str());

    // generate new columns
    this->columnInfos.clear();
    this->columnInfos.resize(outputColumnCount);

    for (int indexX = 0; indexX < outputColumnCount; indexX++) {
        this->columnInfos[indexX]
            .SetName("TSNE" + std::to_string(indexX))
            .SetType(megamol::stdplugin::datatools::table::TableDataCall::ColumnType::QUANTITATIVE)
            .SetMinimumValue(minimas[indexX])
            .SetMaximumValue(maximas[indexX]);
    }

    // Result Matrix into Output
    this->data.clear();
    this->data.reserve(rowsCount * outputColumnCount);


    for (size_t row = 0; row < rowsCount; row++) {
        for (size_t col = 0; col < outputColumnCount; col++)
            this->data.push_back(result[row * outputColumnCount + col]);
    }


    this->dataInHash = inCall->DataHash();
    this->datahash++;
    reduceToNSlot.ResetDirty();
    maxIterSlot.ResetDirty();
    randomSeedSlot.ResetDirty();
    thetaSlot.ResetDirty();
    perplexitySlot.ResetDirty();

    free(result);
    result = NULL;
    free(inputData);
    inputData = NULL;
    delete (tsne);
    delete[] maximas;
    delete[] minimas;

    return true;
}
