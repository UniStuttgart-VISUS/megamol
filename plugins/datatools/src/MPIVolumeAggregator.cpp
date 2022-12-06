/*
 * MPIParticleCollector.h
 *
 * Copyright (C) 2014 by S. Grottel
 * Alle Rechte vorbehalten.
 */
#include "MPIVolumeAggregator.h"
#include "cluster/mpi/MpiCall.h"
#include "geometry_calls/MultiParticleDataCall.h"
#include "mmcore/param/EnumParam.h"
#include "vislib/sys/SystemInformation.h"
#include <chrono>

using namespace megamol;


/*
 * datatools::MPIVolumeAggregator::MPIVolumeAggregator
 */
datatools::MPIVolumeAggregator::MPIVolumeAggregator(void)
        : AbstractVolumeManipulator("outData", "indata")
        , callRequestMpi("requestMpi", "Requests initialisation of MPI and the communicator for the view.")
        , operatorSlot("operator", "the operator to apply to the volume when aggregating") {

    this->callRequestMpi.SetCompatibleCall<core::cluster::mpi::MpiCallDescription>();
    this->MakeSlotAvailable(&this->callRequestMpi);

    auto* ep = new core::param::EnumParam(2);
    ep->SetTypePair(0, "Max");
    ep->SetTypePair(1, "Min");
    ep->SetTypePair(2, "Sum");
    ep->SetTypePair(3, "Product");
    this->operatorSlot << ep;
    this->MakeSlotAvailable(&this->operatorSlot);
}


/*
 * datatools::MPIVolumeAggregator::~MPIVolumeAggregator
 */
datatools::MPIVolumeAggregator::~MPIVolumeAggregator(void) {
    this->Release();
}


/*
 * datatools::MPIVolumeAggregator::manipulateData
 */
bool datatools::MPIVolumeAggregator::manipulateData(
    geocalls::VolumetricDataCall& outData, geocalls::VolumetricDataCall& inData) {
    using geocalls::VolumetricDataCall;

    outData = inData; // also transfers the unlocker to 'outData'

    inData.SetUnlocker(nullptr, false); // keep original data locked
                                        // original data will be unlocked through outData

// without mpi, this module does nothing at all
#ifdef MEGAMOL_USE_MPI
    bool useMpi = initMPI();

    if (!useMpi) {
        return true;
    }
    if (!inData(VolumetricDataCall::IDX_GET_EXTENTS)) {
        megamol::core::utility::log::Log::DefaultLog.WriteInfo("MPIVolumeAggregator: No extents available.\n");
        return false;
    }
    if (!inData(VolumetricDataCall::IDX_GET_METADATA)) {
        megamol::core::utility::log::Log::DefaultLog.WriteInfo("MPIVolumeAggregator: No metadata available.\n");
        return false;
    }

    megamol::core::utility::log::Log::DefaultLog.WriteInfo("MPIVolumeAggregator: starting volume aggregation");
    const auto startAllTime = std::chrono::high_resolution_clock::now();

    metadata = inData.GetMetadata()->Clone();
    const auto comp = metadata.Components;

    if (metadata.GridType != geocalls::CARTESIAN && metadata.GridType != geocalls::RECTILINEAR) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "MPIVolumeAggregator cannot work with grid type %d", metadata.GridType);
        return false;
    }
    if (metadata.ScalarType != geocalls::FLOATING_POINT) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "MPIVolumeAggregator cannot work with scalar type %d", metadata.ScalarType);
        return false;
    }

    const size_t numFloats = comp * metadata.Resolution[0] * metadata.Resolution[1] * metadata.Resolution[2];
    // we need a copy of the data since we must not alter it.
    std::vector<float> tmpVolume;
    tmpVolume.resize(numFloats);
    memcpy(tmpVolume.data(), inData.GetData(), numFloats * sizeof(float));
    // and a copy to receive the result
    this->theVolume.resize(numFloats);

    MPI_Op op = MPI_SUM;
    const auto opVal = this->operatorSlot.Param<core::param::EnumParam>()->Value();
    if (comp > 1) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "MPIVolumeAggregator: multi-component volume detected! Computing min/max density on the first component "
            "only. Op %s is applied on all components.",
            this->operatorSlot.Param<core::param::EnumParam>()->getMap()[opVal].c_str());
    }
    switch (opVal) {
    case 0:
        op = MPI_MAX;
        break;
    case 1:
        op = MPI_MIN;
        break;
    case 2:
        op = MPI_SUM;
        break;
    case 3:
        op = MPI_PROD;
        break;
    default:
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "MPIVolumeAggregator: unknown operation %u. Aborting.", opVal);
        return false;
    }

    megamol::core::utility::log::Log::DefaultLog.WriteInfo("MPIVolumeAggregator: starting Allreduce");
    const auto startTime = std::chrono::high_resolution_clock::now();

    MPI_Allreduce(tmpVolume.data(), this->theVolume.data(), numFloats, MPI_FLOAT, op, this->comm);

    const auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> diffMillis = endTime - startTime;

    const int chunkSize = (numFloats) / this->mpiSize + 1;
    float min = std::numeric_limits<float>::max();
    float max = 0.0f;
    float globalmin = min;
    float globalmax = max;
    const int end = std::min<int>((this->mpiRank + 1) * chunkSize, numFloats);
    for (int x = this->mpiRank * chunkSize; x < end; x += comp) {
        auto& d = this->theVolume.data()[x];
        if (d < min) {
            min = d;
        }
        if (d > max) {
            max = d;
        }
    }

    // now make min max global
    MPI_Allreduce(&min, &globalmin, 1, MPI_FLOAT, MPI_MIN, this->comm);
    MPI_Allreduce(&max, &globalmax, 1, MPI_FLOAT, MPI_MAX, this->comm);

    const auto endAllTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> diffAllMillis = endAllTime - startAllTime;
    megamol::core::utility::log::Log::DefaultLog.WriteInfo(
        "MPIVolumeAggregator: Allreduce of %u x %u x %u volume took %f ms.", metadata.Resolution[0],
        metadata.Resolution[1], metadata.Resolution[2], diffMillis.count());
    megamol::core::utility::log::Log::DefaultLog.WriteInfo(
        "MPIVolumeAggregator: volume aggregation of %u x %u x %u volume took %f ms.", metadata.Resolution[0],
        metadata.Resolution[1], metadata.Resolution[2], diffAllMillis.count());

    outData.SetData(this->theVolume.data());
    metadata.MinValues[0] = globalmin;
    metadata.MaxValues[0] = globalmax;
    outData.SetMetadata(&metadata);
#endif /* MEGAMOL_USE_MPI */

    return true;
}

bool datatools::MPIVolumeAggregator::initMPI() {
    bool retval = false;
#ifdef MEGAMOL_USE_MPI
    if (this->comm == MPI_COMM_NULL) {
        auto c = this->callRequestMpi.CallAs<core::cluster::mpi::MpiCall>();
        if (c != nullptr) {
            /* New method: let MpiProvider do all the stuff. */
            if ((*c)(core::cluster::mpi::MpiCall::IDX_PROVIDE_MPI)) {
                megamol::core::utility::log::Log::DefaultLog.WriteInfo("Got MPI communicator.");
                this->comm = c->GetComm();
            } else {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    _T("Could not ")
                    _T("retrieve MPI communicator for the MPI-based view ")
                    _T("from the registered provider module."));
            }
        }

        if (this->comm != MPI_COMM_NULL) {
            megamol::core::utility::log::Log::DefaultLog.WriteInfo(_T("MPI is ready, ")
                                                                   _T("retrieving communicator properties ..."));
            ::MPI_Comm_rank(this->comm, &this->mpiRank);
            ::MPI_Comm_size(this->comm, &this->mpiSize);
            megamol::core::utility::log::Log::DefaultLog.WriteInfo(_T("This MPIParticleCollector on %hs is %d ")
                                                                   _T("of %d."),
                vislib::sys::SystemInformation::ComputerNameA().PeekBuffer(), this->mpiRank, this->mpiSize);
        } /* end if (this->comm != MPI_COMM_NULL) */
    }     /* end if (this->comm == MPI_COMM_NULL) */

    /* Determine success of the whole operation. */
    retval = (this->comm != MPI_COMM_NULL);
#endif /* MEGAMOL_USE_MPI */
    return retval;
}

void datatools::MPIVolumeAggregator::release() {
    delete[] this->metadata.MinValues;
    delete[] this->metadata.MaxValues;
    delete[] this->metadata.SliceDists[0];
    delete[] this->metadata.SliceDists[1];
    delete[] this->metadata.SliceDists[2];
}
