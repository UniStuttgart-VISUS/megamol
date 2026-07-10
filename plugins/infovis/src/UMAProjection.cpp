/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#include "UMAProjection.h"

#include <sstream>

#include <umappp/umappp.hpp>

#include "datatools/table/TableDataCall.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"

using namespace megamol;
using namespace megamol::infovis;

namespace {
    /// Taken from 
    /// https://github.com/libscran/umappp/blob/4a0ff313f967c794bbfe292891677b4847d94447/include/umappp/Umap.hpp
    /// After this commit the Umap class and Defaults struct have been removed
    struct UmapDefaults {
        using Float = double;

        static constexpr Float local_connectivity = 1.0;

        static constexpr Float bandwidth = 1;

        static constexpr Float mix_ratio = 1;

        static constexpr Float spread = 1;

        static constexpr Float min_dist = 0.01;

        static constexpr Float a = 0;

        static constexpr Float b = 0;

        static constexpr Float repulsion_strength = 1;

        //static constexpr InitMethod initialize = SPECTRAL;

        static constexpr int num_epochs = -1;

        static constexpr Float learning_rate = 1; 

        static constexpr Float negative_sample_rate = 5;

        static constexpr int num_neighbors = 15;

        static constexpr uint64_t seed = 1234567890;

        static constexpr int num_threads = 1;

        static constexpr int parallel_optimization = false;
    };
}

UMAProjection::UMAProjection()
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

    randomSeedSlot << new ::megamol::core::param::IntParam(UmapDefaults::seed);
    this->MakeSlotAvailable(&randomSeedSlot);

    nEpochsSlot << new ::megamol::core::param::IntParam(UmapDefaults::num_epochs);
    this->MakeSlotAvailable(&nEpochsSlot);

    learningRateSlot << new ::megamol::core::param::FloatParam(UmapDefaults::learning_rate);
    this->MakeSlotAvailable(&learningRateSlot);

    localConnectivitySlot << new ::megamol::core::param::FloatParam(UmapDefaults::local_connectivity);
    this->MakeSlotAvailable(&localConnectivitySlot);

    bandwidthSlot << new ::megamol::core::param::FloatParam(UmapDefaults::bandwidth);
    this->MakeSlotAvailable(&bandwidthSlot);


    mixRatioSlot << new ::megamol::core::param::FloatParam(UmapDefaults::mix_ratio);
    this->MakeSlotAvailable(&mixRatioSlot);

    spreadSlot << new ::megamol::core::param::FloatParam(UmapDefaults::spread);
    this->MakeSlotAvailable(&spreadSlot);

    minDistSlot << new ::megamol::core::param::FloatParam(UmapDefaults::min_dist);
    this->MakeSlotAvailable(&minDistSlot);

    aSlot << new ::megamol::core::param::FloatParam(UmapDefaults::a);
    this->MakeSlotAvailable(&aSlot);

    bSlot << new ::megamol::core::param::FloatParam(UmapDefaults::b);
    this->MakeSlotAvailable(&bSlot);

    repulsionStrengthSlot << new ::megamol::core::param::FloatParam(UmapDefaults::repulsion_strength);
    this->MakeSlotAvailable(&repulsionStrengthSlot);

    // problem: enum changed to
    // enum InitializeMethod : char { SPECTRAL, RANDOM, NONE };
    initializeSlot << new ::megamol::core::param::EnumParam(0);
    initializeSlot.Param<param::EnumParam>()->SetTypePair(0, "spectral (fallback: random)");
    initializeSlot.Param<param::EnumParam>()->SetTypePair(1, "spectral (fallback: existing)");
    initializeSlot.Param<param::EnumParam>()->SetTypePair(2, "random");
    //initializeSlot.Param<param::EnumParam>()->SetTypePair(3, "existing");
    this->MakeSlotAvailable(&initializeSlot);

    negativeSampleRateSlot << new ::megamol::core::param::FloatParam(UmapDefaults::negative_sample_rate);
    this->MakeSlotAvailable(&negativeSampleRateSlot);

    nNeighborsSlot << new ::megamol::core::param::IntParam(UmapDefaults::num_neighbors);
    this->MakeSlotAvailable(&nNeighborsSlot);
}

UMAProjection::~UMAProjection() {
    this->Release();
}

bool UMAProjection::create() {
    return true;
}

void UMAProjection::release() {}

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

    auto dimCount = inCall->GetColumnsCount();
    auto obsCount = inCall->GetRowsCount();
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

    // Transform row-major to column-major format.
    std::vector<double> inputData(dimCount * obsCount, 0.0);
    for (int dim = 0; dim < dimCount; dim++) {
        for (int obs = 0; obs < obsCount; obs++) {
            inputData[dim * obsCount + obs] = inData[obs * dimCount + dim];
        }
    }

    // Allocate a column-major embedding array.
    std::vector<double> embeddingData(nDims * obsCount, 0.0);

    // Run UMAP algorithm.
    umappp::Options opt;
    opt.optimize_seed = randomSeed;
    opt.num_epochs = nEpochs;
    opt.learning_rate = learningRate;
    opt.local_connectivity = localConnectivity;
    opt.bandwidth = bandwidth;
    opt.mix_ratio = mixRatio;
    opt.spread = spread;
    opt.min_dist = minDist;
    opt.a = a;
    opt.b = b;
    opt.repulsion_strength = repulsionStrength;
    opt.negative_sample_rate = negativeSampleRate;
    opt.num_neighbors = nNeighbors;

    opt.initialize_method = static_cast<umappp::InitializeMethod>(initialize);

    knncolle::VptreeBuilder<int, double, double> vp_builder(
        std::make_shared<knncolle::EuclideanDistance<double, double> >()
    );

    auto status = umappp::initialize(
        dimCount, 
        (int)obsCount, 
        inputData.data(), 
        vp_builder, 
        nDims, 
        embeddingData.data(), 
        opt);

    status.run(embeddingData.data());

    megamol::core::utility::log::Log::DefaultLog.WriteInfo(_T("Epoch %d of %d; a: %lf b: %lf, obs: %d\n"),
        status.epoch(), 
        status.num_epochs(), 
        opt.a.value(), 
        opt.b.value(), 
        status.num_observations()
    );

    // Search extreme values.
    std::vector<double> minimas(nDims, 0.0);
    std::vector<double> maximas(nDims, 0.0);
    for (int dim = 0; dim < nDims; dim++) {
        minimas[dim] = maximas[dim] = embeddingData[dim * obsCount];
    }
    for (int dim = 0; dim < nDims; dim++) {
        for (int obs = 1; obs < obsCount; obs++) {
            auto value = embeddingData[dim * obsCount + obs];
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
    this->data.reserve(obsCount * nDims);
    for (int obs = 0; obs < obsCount; obs++) {
        for (int dim = 0; dim < nDims; dim++) {
            this->data.push_back(static_cast<float>(embeddingData[dim * obsCount + obs]));
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
