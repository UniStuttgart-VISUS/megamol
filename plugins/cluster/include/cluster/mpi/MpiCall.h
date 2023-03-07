/*
 * MpiCall.h
 *
 * Copyright (C) 2014 Visualisierungsinstitut der Universität Stuttgart.
 * Alle Rechte vorbehalten.
 */

#pragma once

#ifdef MEGAMOL_USE_MPI
#include <mpi.h>
#endif /* MEGAMOL_USE_MPI */

#include "mmcore/Call.h"
#include "mmcore/factories/CallAutoDescription.h"


namespace megamol::core::cluster::mpi {

/**
 * This call requests MpiProvider to initialise MPI and to return a
 * communicator that a partition of the comm world can use.
 */
class MpiCall : public Call {

public:
    /**
     * Answer the name of this call.
     *
     * @return The name of this call.
     */
    static inline const char* ClassName() {
        return "MpiCall";
    }

    /**
     * Answer a human readable description of this call.
     *
     * @return A human readable description of this call.
     */
    static inline const char* Description() {
        return "Requests lazy initalisation of MPI.";
    }

    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount();

    /**
     * Answer the name of the function used for this call.
     *
     * @param idx The index of the function to return it's name.
     *
     * @return The name of the requested function.
     */
    static const char* FunctionName(unsigned int idx);

    /**
     * Answers whether this call is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable();

    /** Index of the intent initialising MPI. */
    static const unsigned int IDX_PROVIDE_MPI;

    /**
     * Initialises a new instance.
     */
    MpiCall();

    /**
     * Finalises the instance.
     */
    ~MpiCall() override;

#ifdef MEGAMOL_USE_MPI
    /**
     * Answer the communicator created during node colouring.
     *
     * @return The communicator that the caller should use.
     */
    inline MPI_Comm GetComm(void) const {
        return this->comm;
    }
#endif /* MEGAMOL_USE_MPI */

    /**
     * Get the size of the communicator that has been created during node
     * colouring.
     *
     * @return The size of the communicator.
     */
    int GetCommSize() const;

    /**
     * Get the rank of the calling process in the communicator that has been
     * created during node colouring.
     *
     * @return The rank of the calling process.
     */
    int GetRank() const;

#ifdef MEGAMOL_USE_MPI
    /**
     * Set the communicator created during node colouring.
     *
     * @param comm The communicator that the caller should use.
     */
    inline void SetComm(const MPI_Comm comm) {
        this->comm = comm;
    }
#endif /* MEGAMOL_USE_MPI */

private:
    /** Super class. */
    typedef Call Base;

    /** The intents that are provided by the call. */
    static const char* INTENTS[1];

#ifdef MEGAMOL_USE_MPI
    /** The communicator that the caller should use. */
    MPI_Comm comm;
#endif /* MEGAMOL_USE_MPI */
};

typedef factories::CallAutoDescription<MpiCall> MpiCallDescription;

} // namespace megamol::core::cluster::mpi
