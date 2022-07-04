#include "UMAProjection.h"

#include "datatools/table/TableDataCall.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"

#include <sstream>
#include <umappp/Umap.hpp>

using namespace megamol;
using namespace megamol::infovis;

using Umap = umappp::Umap<double>;

UMAProjection::UMAProjection(void)
        : megamol::core::Module()
        , dataOutSlot("dataOut", "Ouput")
        , dataInSlot("dataIn", "Input")
        , nDimsSlot("nDims", "Number of dimensions to keep")
        , randomSeedSlot("randomSeed", "Seed to use for the Mersenne Twister when sampling negative observations.")
        , nEpochsSlot("nEpochs", "Number of epochs for the gradient descent, i.e., optimization iterations. Larger "
                                 "values improve convergence at the cost of computational work.")
        , learningRateSlot("learningRate", "Initial learning rate used in the gradient descent. Larger values can "
                                           "improve the speed of convergence but at the cost of stability.")
        , localConnectivitySlot("localConnectivity",
              "The number of nearest neighbors that are assumed to be always connected, with "
              "maximum membership confidence. Larger values increase the connectivity of the "
              "embedding and reduce the focus on local structure.")
        , bandwidthSlot("bandwidth",
              "Effective bandwidth of the kernel when converting the "
              "distance to a neighbor into a fuzzy "
              "set membership confidence. Larger values reduce the decay in confidence with respect to "
              "distance, increasing connectivity and favoring global structure. ")
        , mixRatioSlot("mixRatio",
              "Mixing ratio to use when combining fuzzy sets. This "
              "symmetrizes the sets by ensuring that "
              "the confidence of $a$ belonging to $b$'s set is the same as the confidence of $b$ belonging "
              "to $a$'s set. A mixing ratio of 1 will take the union of confidences, a ratio of 0 will "
              "take the intersection, and intermediate values will interpolate between them. Larger values "
              "(up to 1) favor connectivity and more global structure.")
        , spreadSlot("spread", "Scale of the coordinates of the final low-dimensional embedding.")
        , minDistSlot("minDist", "Minimum distance between observations in the final low-dimensional embedding. "
                                 "Smaller values will increase local clustering while larger values favors a more even "
                                 "distribution.This is interpreted relative to the spread of points in spread")
        , aSlot("a", "Positive value for the $a$ parameter for the fuzzy set membership strength calculations. Larger "
                     "values yield a sharper decay in membership strength with increasing distance between "
                     "observations. Inferred if set to zero.")
        , bSlot("b", "Value in $(0, 1)$ for the $b$ parameter for the fuzzy set membership strength calculations. "
                     "Larger values yield an earlier decay in membership strength with increasing distance between "
                     "observations. Inferrred if set to zero.")
        , repulsionStrengthSlot("repulsionStrength",
              "Modifier for the repulsive force. Larger values increase repulsion and favor local structure.")
        , initializeSlot("initialize", "How to initialize the embedding.")
        , negativeSampleRateSlot("negativeSampleRate",
              "Rate of sampling negative observations to compute repulsive forces. This is interpreted with respect to "
              "the number of neighbors with attractive forces, i.e., for each attractive interaction, `n` negative "
              "samples are taken for repulsive interactions. Smaller values can improve the speed of convergence but "
              "at the cost of stability.")
        , nNeighborsSlot("nNeighbors",
              "Number of neighbors to use to define the fuzzy sets. Larger values improve connectivity "
              "and favor preservation of global structure, at the cost of increased computational "
              "work. Only used when identifying nearest neighbors.")
        , datahash(0)
        , dataInHash(0)
        , columnInfos() {

    this->dataInSlot.SetCompatibleCall<megamol::datatools::table::TableDataCallDescription>();
    this->MakeSlotAvailable(&this->dataInSlot);

    this->dataOutSlot.SetCallback(megamol::datatools::table::TableDataCall::ClassName(),
        megamol::datatools::table::TableDataCall::FunctionName(0), &UMAProjection::getDataCallback);
    this->dataOutSlot.SetCallback(megamol::datatools::table::TableDataCall::ClassName(),
        megamol::datatools::table::TableDataCall::FunctionName(1), &UMAProjection::getHashCallback);
    this->MakeSlotAvailable(&this->dataOutSlot);

    nDimsSlot << new ::megamol::core::param::IntParam(2);
    this->MakeSlotAvailable(&nDimsSlot);

    randomSeedSlot << new ::megamol::core::param::IntParam(Umap::Defaults::seed);
    this->MakeSlotAvailable(&randomSeedSlot);

    nEpochsSlot << new ::megamol::core::param::IntParam(Umap::Defaults::num_epochs);
    this->MakeSlotAvailable(&nEpochsSlot);

    learningRateSlot << new ::megamol::core::param::FloatParam(Umap::Defaults::learning_rate);
    this->MakeSlotAvailable(&learningRateSlot);

    localConnectivitySlot << new ::megamol::core::param::FloatParam(Umap::Defaults::local_connectivity);
    this->MakeSlotAvailable(&localConnectivitySlot);

    bandwidthSlot << new ::megamol::core::param::FloatParam(Umap::Defaults::bandwidth);
    this->MakeSlotAvailable(&bandwidthSlot);


    mixRatioSlot << new ::megamol::core::param::FloatParam(Umap::Defaults::mix_ratio);
    this->MakeSlotAvailable(&mixRatioSlot);

    spreadSlot << new ::megamol::core::param::FloatParam(Umap::Defaults::spread);
    this->MakeSlotAvailable(&spreadSlot);

    minDistSlot << new ::megamol::core::param::FloatParam(Umap::Defaults::min_dist);
    this->MakeSlotAvailable(&minDistSlot);

    aSlot << new ::megamol::core::param::FloatParam(Umap::Defaults::a);
    this->MakeSlotAvailable(&aSlot);

    bSlot << new ::megamol::core::param::FloatParam(Umap::Defaults::b);
    this->MakeSlotAvailable(&bSlot);

    repulsionStrengthSlot << new ::megamol::core::param::FloatParam(Umap::Defaults::repulsion_strength);
    this->MakeSlotAvailable(&repulsionStrengthSlot);

    initializeSlot << new ::megamol::core::param::EnumParam(0);
    initializeSlot.Param<param::EnumParam>()->SetTypePair(0, "spectral (fallback: random)");
    initializeSlot.Param<param::EnumParam>()->SetTypePair(1, "spectral (fallback: existing)");
    initializeSlot.Param<param::EnumParam>()->SetTypePair(2, "random");
    //initializeSlot.Param<param::EnumParam>()->SetTypePair(3, "existing");
    this->MakeSlotAvailable(&initializeSlot);

    negativeSampleRateSlot << new ::megamol::core::param::FloatParam(Umap::Defaults::negative_sample_rate);
    this->MakeSlotAvailable(&negativeSampleRateSlot);

    nNeighborsSlot << new ::megamol::core::param::IntParam(Umap::Defaults::num_neighbors);
    this->MakeSlotAvailable(&nNeighborsSlot);
}

UMAProjection::~UMAProjection(void) {
    this->Release();
}

bool UMAProjection::create(void) {
    return true;
}

void UMAProjection::release(void) {}

bool UMAProjection::getDataCallback(core::Call& c) {
    try {
        megamol::datatools::table::TableDataCall* outCall = dynamic_cast<megamol::datatools::table::TableDataCall*>(&c);
        if (outCall == NULL)
            return false;

        megamol::datatools::table::TableDataCall* inCall =
            this->dataInSlot.CallAs<megamol::datatools::table::TableDataCall>();
        if (inCall == NULL)
            return false;

        inCall->SetFrameID(outCall->GetFrameID());
        if (!(*inCall)())
            return false;

        bool finished = project(inCall);
        if (finished == false)
            return false;

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
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            _T("Failed to execute %hs::processData\n"), ClassName());
        return false;
    }

    return true;
}

bool UMAProjection::getHashCallback(core::Call& c) {
    try {
        megamol::datatools::table::TableDataCall* outCall = dynamic_cast<megamol::datatools::table::TableDataCall*>(&c);
        if (outCall == NULL)
            return false;

        megamol::datatools::table::TableDataCall* inCall =
            this->dataInSlot.CallAs<megamol::datatools::table::TableDataCall>();
        if (inCall == NULL)
            return false;

        inCall->SetFrameID(outCall->GetFrameID());
        if (!(*inCall)(1))
            return false;

        outCall->SetFrameCount(inCall->GetFrameCount());
        outCall->SetDataHash(this->datahash);
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            _T("Failed to execute %hs::getHashCallback\n"), ClassName());
        return false;
    }

    return true;
}

bool megamol::infovis::UMAProjection::project(megamol::datatools::table::TableDataCall* inCall) {
    // Check if input data or slots have changed
    if (this->dataInHash == inCall->DataHash()) {
        if (!nDimsSlot.IsDirty() && !randomSeedSlot.IsDirty() && !nEpochsSlot.IsDirty() &&
            !learningRateSlot.IsDirty() && !localConnectivitySlot.IsDirty() && !bandwidthSlot.IsDirty() &&
            !mixRatioSlot.IsDirty() && !spreadSlot.IsDirty() && !minDistSlot.IsDirty() && !aSlot.IsDirty() &&
            !bSlot.IsDirty() && !repulsionStrengthSlot.IsDirty() && !initializeSlot.IsDirty() &&
            !negativeSampleRateSlot.IsDirty() && !nNeighborsSlot.IsDirty()) {
            return true; // Nothing to do
        }
    }

    auto columnCount = inCall->GetColumnsCount();
    auto columnInfos = inCall->GetColumnsInfos();
    auto rowsCount = inCall->GetRowsCount();
    auto inData = inCall->GetData();

    // Fetch parameters.
    auto nDims = this->nDimsSlot.Param<core::param::IntParam>()->Value();
    auto randomSeed = this->randomSeedSlot.Param<core::param::IntParam>()->Value();
    auto nEpochs = this->nEpochsSlot.Param<core::param::IntParam>()->Value();
    auto learningRate = this->learningRateSlot.Param<core::param::FloatParam>()->Value();
    auto localConnectivity = this->localConnectivitySlot.Param<core::param::FloatParam>()->Value();
    auto bandwidth = this->bandwidthSlot.Param<core::param::FloatParam>()->Value();
    auto mixRatio = this->mixRatioSlot.Param<core::param::FloatParam>()->Value();
    auto spread = this->spreadSlot.Param<core::param::FloatParam>()->Value();
    auto minDist = this->minDistSlot.Param<core::param::FloatParam>()->Value();
    auto a = this->aSlot.Param<core::param::FloatParam>()->Value();
    auto b = this->bSlot.Param<core::param::FloatParam>()->Value();
    auto repulsionStrength = this->repulsionStrengthSlot.Param<core::param::FloatParam>()->Value();
    auto initialize = this->initializeSlot.Param<core::param::EnumParam>()->Value();
    auto negativeSampleRate = this->negativeSampleRateSlot.Param<core::param::FloatParam>()->Value();
    auto nNeighbors = this->nNeighborsSlot.Param<core::param::IntParam>()->Value();

    // Load data in a column-major input array.
    std::vector<double> inputData(columnCount * rowsCount, 0.0);
    for (int col = 0; col < columnCount; col++) {
        for (int row = 0; row < rowsCount; row++) {
            inputData[col * rowsCount + row] = inData[row * columnCount + col];
        }
    }

    // Allocate a column-major embedding array.
    std::vector<double> embeddingData(nDims * rowsCount, 0.0);

    // Run UMAP algorithm.
    Umap umap;
    umap.set_seed(randomSeed);
    umap.set_num_epochs(nEpochs);
    umap.set_learning_rate(learningRate);
    umap.set_local_connectivity(localConnectivity);
    umap.set_bandwidth(bandwidth);
    umap.set_mix_ratio(mixRatio);
    umap.set_spread(spread);
    umap.set_min_dist(minDist);
    umap.set_a(a);
    umap.set_b(b);
    umap.set_repulsion_strength(repulsionStrength);
    umap.set_initialize(static_cast<umappp::InitMethod>(initialize));
    umap.set_negative_sample_rate(negativeSampleRate);
    umap.set_num_neighbors(nNeighbors);
    auto status = umap.run(columnCount, rowsCount, inputData.data(), nDims, embeddingData.data(), 0);
    megamol::core::utility::log::Log::DefaultLog.WriteInfo(
        _T("Epoch %d of %d; a: %lf b: %lf, obs: %d\n"),
        status.epoch(), status.num_epochs(), status.a, status.b, status.nobs());

    // Search extreme values.
    std::vector<double> minimas(nDims, 0.0);
    std::vector<double> maximas(nDims, 0.0);
    for (int dim = 0; dim < nDims; dim++) {
        minimas[dim] = maximas[dim] = embeddingData[dim * rowsCount];
    }
    for (int dim = 0; dim < nDims; dim++) {
        for (int obs = 1; obs < rowsCount; obs++) {
            auto value = embeddingData[dim * rowsCount + obs];
            if (maximas[dim] < value)
                maximas[dim] = value;
            if (minimas[dim] > value)
                minimas[dim] = value;
        }
    }

    // Generate output column infos.
    this->columnInfos.clear();
    this->columnInfos.resize(nDims);
    for (int dim = 0; dim < nDims; dim++) {
        this->columnInfos[dim]
            .SetName("umap" + std::to_string(dim))
            .SetType(megamol::datatools::table::TableDataCall::ColumnType::QUANTITATIVE)
            .SetMinimumValue(minimas[dim])
            .SetMaximumValue(maximas[dim]);
    }

    // Copy embedding to output.
    this->data.clear();
    this->data.reserve(rowsCount * nDims);
    for (int obs = 0; obs < rowsCount; obs++) {
        for (int dim = 0; dim < nDims; dim++) {
            this->data.push_back(embeddingData[dim * rowsCount + obs]);
        }
    }

    this->dataInHash = inCall->DataHash();
    this->datahash++;
    nDimsSlot.ResetDirty();
    randomSeedSlot.ResetDirty();
    nEpochsSlot.ResetDirty();
    learningRateSlot.ResetDirty();
    localConnectivitySlot.ResetDirty();
    bandwidthSlot.ResetDirty();
    mixRatioSlot.ResetDirty();
    spreadSlot.ResetDirty();
    minDistSlot.ResetDirty();
    aSlot.ResetDirty();
    bSlot.ResetDirty();
    repulsionStrengthSlot.ResetDirty();
    initializeSlot.ResetDirty();
    negativeSampleRateSlot.ResetDirty();
    nNeighborsSlot.ResetDirty();

    return true;
}
