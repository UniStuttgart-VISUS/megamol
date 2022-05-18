/*
 * MpiCall.cpp
 *
 * Copyright (C) 2014 Visualisierungsinstitut der Universität Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "mmcore/cluster/mpi/MpiCall.h"
#include "stdafx.h"

#include "mmcore/utility/log/Log.h"
#include "vislib/sys/CmdLineProvider.h"


/*
 * megamol::core::cluster::mpi::MpiCall::FunctionCount
 */
unsigned int megamol::core::cluster::mpi::MpiCall::FunctionCount(void) {
    return (sizeof(MpiCall::INTENTS) / sizeof(*MpiCall::INTENTS));
}


/*
 * megamol::core::cluster::mpi::MpiCall::FunctionName
 */
const char* megamol::core::cluster::mpi::MpiCall::FunctionName(unsigned int idx) {
    if (idx < MpiCall::FunctionCount()) {
        return MpiCall::INTENTS[idx];
    } else {
        return "";
    }
}


/*
 * megamol::core::cluster::mpi::MpiCall::IsAvailable
 */
bool megamol::core::cluster::mpi::MpiCall::IsAvailable(void) {
#ifdef WITH_MPI
    return true;
#else  /* WITH_MPI */
    return false;
#endif /* WITH_MPI */
}


/*
 * megamol::core::cluster::mpi::MpiCall::IDX_PROVIDE_MPI
 */
const unsigned int megamol::core::cluster::mpi::MpiCall::IDX_PROVIDE_MPI = 0;


/*
 * megamol::core::cluster::mpi::MpiCall::MpiCall
 */
megamol::core::cluster::mpi::MpiCall::MpiCall(void) : Base() {
#ifdef WITH_MPI
    this->comm = MPI_COMM_NULL;
#endif /* WITH_MPI */
}


/*
 * megamol::core::cluster::mpi::MpiCall::~MpiCall
 */
megamol::core::cluster::mpi::MpiCall::~MpiCall(void) {}


/*
 * megamol::core::cluster::mpi::MpiCall::GetCommSize
 */
int megamol::core::cluster::mpi::MpiCall::GetCommSize(void) const {
    int retval = -1;
#ifdef WITH_MPI
    ::MPI_Comm_size(this->comm, &retval);
#endif /* WITH_MPI */
    return retval;
}


/*
 * megamol::core::cluster::mpi::MpiCall::GetRank
 */
int megamol::core::cluster::mpi::MpiCall::GetRank(void) const {
    int retval = -1;
#ifdef WITH_MPI
    ::MPI_Comm_rank(this->comm, &retval);
#endif /* WITH_MPI */
    return retval;
}


/*
 * megamol::core::misc::VolumetricDataCall::INTENTS
 */
const char* megamol::core::cluster::mpi::MpiCall::INTENTS[] = {
    "ProvideMpi",
};
