/*
 * MPIParticleCollector.h
 *
 * Copyright (C) 2014 by S. Grottel
 * Alle Rechte vorbehalten.
 */
#include "MPIParticleCollector.h"
#include "cluster/mpi/MpiCall.h"
#include "vislib/sys/SystemInformation.h"

using namespace megamol;


/*
 * datatools::MPIParticleCollector::MPIParticleCollector
 */
datatools::MPIParticleCollector::MPIParticleCollector(void)
        : AbstractParticleManipulator("outData", "indata")
        , callRequestMpi("requestMpi", "Requests initialisation of MPI and the communicator for the view.") {

    this->callRequestMpi.SetCompatibleCall<core::cluster::mpi::MpiCallDescription>();
    this->MakeSlotAvailable(&this->callRequestMpi);
}


/*
 * datatools::MPIParticleCollector::~MPIParticleCollector
 */
datatools::MPIParticleCollector::~MPIParticleCollector(void) {
    this->Release();
}


/*
 * datatools::MPIParticleCollector::manipulateData
 */
bool datatools::MPIParticleCollector::manipulateData(
    geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData) {
    using geocalls::MultiParticleDataCall;

    outData = inData; // also transfers the unlocker to 'outData'

    inData.SetUnlocker(nullptr, false); // keep original data locked
                                        // original data will be unlocked through outData
#ifdef WITH_MPI
    bool useMpi = initMPI();

    unsigned int plc = outData.GetParticleListCount();
    for (unsigned int i = 0; i < plc; i++) {
        MultiParticleDataCall::Particles& p = outData.AccessParticles(i);

        uint64_t cnt = p.GetCount();

        const uint8_t* cd = reinterpret_cast<const uint8_t*>(p.GetColourData());
        unsigned int cds = p.GetColourDataStride();
        MultiParticleDataCall::Particles::ColourDataType cdt = p.GetColourDataType();
        unsigned int csize = MultiParticleDataCall::Particles::ColorDataSize[cdt];

        const uint8_t* vd = reinterpret_cast<const uint8_t*>(p.GetVertexData());
        unsigned int vds = p.GetVertexDataStride();
        MultiParticleDataCall::Particles::VertexDataType vdt = p.GetVertexDataType();
        unsigned int vsize = MultiParticleDataCall::Particles::VertexDataSize[vdt];

        //megamol::core::utility::log::Log::DefaultLog.WriteInfo("Count = %lu", cnt);
        //megamol::core::utility::log::Log::DefaultLog.WriteInfo("csize = %u, vsize = %u", csize, vsize);

        std::vector<uint64_t> counts;
        std::vector<int32_t> vertSizes, colSizes, vertOffsets, colOffsets;
        counts.resize(this->mpiSize);
        vertSizes.resize(this->mpiSize);
        colSizes.resize(this->mpiSize);
        vertOffsets.resize(this->mpiSize);
        vertOffsets[0] = 0;
        colOffsets.resize(this->mpiSize);
        colOffsets[0] = 0;
        uint64_t allCount = 0;
        // MPI_Reduce(&cnt, &allCount, 1, MPI_INT64_T, MPI_SUM, 0, this->comm);
        MPI_Gather(&cnt, 1, MPI_UINT64_T, counts.data(), 1, MPI_UINT64_T, 0, this->comm);

        if (this->mpiRank == 0) {
            for (auto x = 0; x < this->mpiSize; ++x) {
                allCount += counts[x];
                vertSizes[x] = counts[x] * vsize;
                colSizes[x] = counts[x] * csize;
                if (x > 0) {
                    vertOffsets[x] = vertOffsets[x - 1] + counts[x - 1] * vsize;
                    colOffsets[x] = colOffsets[x - 1] + counts[x - 1] * csize;
                }
            }
            //megamol::core::utility::log::Log::DefaultLog.WriteInfo("allCount = %lu", allCount);
            if (allCount * csize > std::numeric_limits<int>::max() ||
                allCount * vsize > std::numeric_limits<int>::max()) {
                // this is so bad there is no way to fix it
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "allCount is more powerful than MPI, kill me now. Try subsampling more aggressively", allCount);
                abort();
            }
            p.SetCount(allCount);
            allVertexData.resize(allCount * vsize);
            allColorData.resize(allCount * csize);
        } else {
            p.SetCount(cnt);
        }
        vertexData.resize(cnt * vsize);
        colorData.resize(cnt * csize);

        megamol::core::utility::log::Log::DefaultLog.FlushLog();

#pragma omp parallel for
        for (long long idx = 0; idx < cnt; ++idx) {
            memcpy(colorData.data() + csize * idx, cd + cds * idx, csize);
            memcpy(vertexData.data() + vsize * idx, vd + vds * idx, vsize);
        }

        MPI_Gatherv(colorData.data(), cnt * csize, MPI_BYTE, allColorData.data(), colSizes.data(), colOffsets.data(),
            MPI_BYTE, 0, this->comm);
        MPI_Gatherv(vertexData.data(), cnt * vsize, MPI_BYTE, allVertexData.data(), vertSizes.data(),
            vertOffsets.data(), MPI_BYTE, 0, this->comm);

        if (this->mpiRank == 0) {
            p.SetColourData(cdt, allColorData.data(), csize);
            p.SetVertexData(vdt, allVertexData.data(), vsize);
        } else {
            p.SetColourData(cdt, colorData.data(), csize);
            p.SetVertexData(vdt, vertexData.data(), vsize);
        }
    }
#endif /* WITH_MPI */

    return true;
}

bool datatools::MPIParticleCollector::initMPI() {
    bool retval = false;
#ifdef WITH_MPI
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
#endif /* WITH_MPI */
    return retval;
}
