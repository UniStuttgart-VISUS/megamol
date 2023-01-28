/*
 * MpiProvider.h
 *
 * Copyright (C) 2014 Visualisierungsinstitut der Universität Stuttgart.
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CLUSTER_MPI_MPIPROVIDER_H_INCLUDED
#define MEGAMOLCORE_CLUSTER_MPI_MPIPROVIDER_H_INCLUDED
#pragma once

#ifdef MEGAMOL_USE_MPI
#include <mpi.h>
#endif /* MEGAMOL_USE_MPI */

#include <atomic>
#include <memory>

#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"

#include "mmcore/param/ParamSlot.h"

#include "vislib/String.h"


namespace megamol::core::cluster::mpi {

/**
 * This module lazily initialises MPI and provides the communicator for the
 * MegaMol display nodes.
 */
class MpiProvider : public Module {

public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static inline const char* ClassName() {
        return "MpiProvider";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static inline const char* Description() {
        return "Initialises MPI on behalf of another module.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable();

    /**
     * Initialises a new instance.
     */
    MpiProvider();

    /**
     * Finalises the instance.
     */
    ~MpiProvider() override;

protected:
    /**
     * Initialises the module.
     *
     * @return true unconditionally.
     */
    bool create() override;

    /**
     * Lazily initialises MPI and returns the communicator.
     *
     * @param call
     *
     * @return
     */
    bool OnCallProvideMpi(Call& call);

    /**
     * Finalises the module and releases MPI.
     */
    void release() override;

private:
    /** Super class. */
    typedef Module Base;

    /**
     * Answer the command line the application was started with.
     *
     * @return The command line string.
     */
    static vislib::StringA getCommandLine();

    /**
     * Initialises MPI and performs node colouring.
     *
     * This method performs all necessary checks for MPI being already
     * initialised, "ownership" of the MPI initialisation and node
     * colouring.
     *
     * @param colour The node colour of the calling process.
     *
     * @return true in case of success, false otherwise.
     */
    bool initialiseMpi(const int colour);

    /** Call for retrieving the communicator and other MPI-related data. */
    CalleeSlot callProvideMpi;

    /** Configures the node colour of the MegaMol nodes. */
    param::ParamSlot paramNodeColour;

    /**
     * The number of instances of MpiProvider that are between create() and
     * release() in their lifecycle. These are the instances that might use
     * MPI. If it becomes zero, MPI can be released.
     */
    static std::atomic<int> activeInstances;

    /**
     * Remembers the node colour that was used when initialising the nodes.
     * MPI_UNDEFINED indicates that node colouring has not yet been
     * performed.
     *
     * This atomic also serves as lock that prevents multiple
     * initialisations (enter of critical section).
     */
    std::atomic<int> activeNodeColour;

#ifdef MEGAMOL_USE_MPI
    /**
     * The communicator that was retrieved during node colouring. If this
     * is MPI_COMM_NULL, it indicates that node colouring has not yet been
     * performed.
     */
    std::atomic<MPI_Comm> comm;
#endif /* MEGAMOL_USE_MPI */

    /**
     * Remembers whether the MpiProvider has initialised MPI. In this case,
     * it will also finalise it if the last one was destroyed. Otherwise,
     * someone else is responsible for doing so.
     */
    // static bool isMpiOwner;
};

} // namespace megamol::core::cluster::mpi

#endif /* MEGAMOLCORE_CLUSTER_MPI_MPIPROVIDER_H_INCLUDED */
