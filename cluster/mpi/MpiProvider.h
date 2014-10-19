/*
 * MpiProvider.h
 *
 * Copyright (C) 2014 Visualisierungsinstitut der Universität Stuttgart.
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CLUSTER_MPI_MPIPROVIDER_H_INCLUDED
#define MEGAMOLCORE_CLUSTER_MPI_MPIPROVIDER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#ifdef WITH_MPI
#include <mpi.h>
#endif /* WITH_MPI */

#include "CalleeSlot.h"
#include "Module.h"

#include "param/ParamSlot.h"

#include "vislib/String.h"


namespace megamol {
namespace core {
namespace cluster {
namespace mpi {

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
        static inline const char *ClassName(void) {
            return "MpiProvider";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static inline const char *Description(void) {
            return "Initialises MPI on behalf of another module.";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void);

        /**
         * Disallow usage in quickstarts.
         *
         * @return false
         */
        static bool SupportQuickstart(void) {
            return false;
        }

        /**
         * Initialises a new instance.
         */
        MpiProvider(void);

        /**
         * Finalises the instance.
         */
        virtual ~MpiProvider(void);

    protected:

        /**
         * Initialises the module.
         *
         * @return true unconditionally.
         */
        virtual bool create(void);

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
        virtual void release(void);

    private:

        /** Super class. */
        typedef Module Base;

        /**
         * Answer the command line the application was started with.
         *
         * @return The command line string.
         */
        static vislib::StringA getCommandLine(void);

        /**
         * Initialises MPI and performs node colouring.
         *
         * This method must only be called once, ie if 'comm' was not
         * successfully initialised before. If you violate this precondition,
         * node colouring will be performed anyways.
         *
         * @param colour The node colour of the calling process.
         *
         * @return true in case of success, false otherwise.
         */
        bool initialiseMpi(const int colour);

        /** Call for retrieving the communicator and other MPI-related data. */
        CalleeSlot callProvideMpi;

#ifdef WITH_MPI
        /** The communicator that was retrieved during node colouring. */
        MPI_Comm comm;
#endif /* WITH_MPI */
    };

} /* end namespace mpi */
} /* end namespace cluster */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CLUSTER_MPI_MPIPROVIDER_H_INCLUDED */
