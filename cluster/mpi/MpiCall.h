/*
 * MpiCall.h
 *
 * Copyright (C) 2014 Visualisierungsinstitut der Universität Stuttgart.
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CLUSTER_MPI_MPICALL_H_INCLUDED
#define MEGAMOLCORE_CLUSTER_MPI_MPICALL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#ifdef WITH_MPI
#include <mpi.h>
#endif /* WITH_MPI */

#include "Call.h"
#include "CallAutoDescription.h"


namespace megamol {
namespace core {
namespace cluster {
namespace mpi {

    /**
     * This call requests MpiProvider to initialise MPI and to return a
     * communicator that a partition of the comm world can use.
     *
     * The caller should specify the colour of the calling process for node
     * colouring. Otherwise, the default colour (0) is used.
     *
     * All processes must peform node colouring, that is if a process does not
     * use MpiCall/MpiProvider, it must manually call MPI_Comm_split() on
     * MPI_COMM_WORLD.
     */
    class MEGAMOLCORE_API MpiCall : public Call {

    public:

        /**
         * Answer the name of this call.
         *
         * @return The name of this call.
         */
        static inline const char *ClassName(void) {
            return "MpiCall";
        }

        /**
         * Answer a human readable description of this call.
         *
         * @return A human readable description of this call.
         */
        static inline const char *Description(void) {
            return "Requests lazy initalisation of MPI.";
        }

        /**
         * Answer the number of functions used for this call.
         *
         * @return The number of functions used for this call.
         */
        static unsigned int FunctionCount(void);

        /**
         * Answer the name of the function used for this call.
         *
         * @param idx The index of the function to return it's name.
         *
         * @return The name of the requested function.
         */
        static const char *FunctionName(unsigned int idx);

        /**
         * Answers whether this call is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void);

        /** Index of the intent initialising MPI. */
        static const unsigned int IDX_PROVIDE_MPI;

        /**
         * Initialises a new instance.
         */
        MpiCall(void);

        /**
         * Finalises the instance.
         */
        virtual ~MpiCall(void);

#ifdef WITH_MPI
        /**
         * Answer the communicator created during node colouring.
         *
         * @return The communicator that the caller should use.
         */
        inline MPI_Comm GetComm(void) const {
            return this->comm;
        }
#endif /* WITH_MPI */

        /**
         * Gets the node colour for the calling process.
         *
         * @return The colour for the calling process.
         */
        inline int GetColour(void) const {
            return this->colour;
        }

        /**
         * Get the size of the communicator that has been created during node
         * colouring.
         *
         * @return The size of the communicator.
         */
        int GetCommSize(void) const;

        /**
         * Answer whether MPI was actually initialised during this call.
         *
         * @return true if MPI was initialised during this call, false if it was
         *         initialised before or initialisation failed.
         */
        inline bool GetIsInitialising(void) const {
            return this->isInitialising;
        }

        /**
         * Get the rank of the calling process in the communicator that has been
         * created during node colouring.
         *
         * @return The rank of the calling process.
         */
        int GetRank(void) const;

#ifdef WITH_MPI
        /**
         * Set the communicator created during node colouring.
         *
         * @param comm The communicator that the caller should use.
         */
        inline void SetComm(const MPI_Comm comm) {
            this->comm = comm;
        }
#endif /* WITH_MPI */

        /**
         * Sets the node colour for the calling process that is used when
         * performing node colouring.
         *
         * If no specific colour is set, the node colour is 0.
         *
         * The node colour has no effect if node colouring has already been
         * performed, ie the colour is only used in the first call.
         *
         * @param colour The colour to be used in node colouring.
         */
        inline void SetColour(const int colour) {
            this->colour = colour;
        }

        /**
         * Set whether MPI was initialised during this call.
         *
         * @param isInitialising
         */
        inline void SetIsInitialising(const bool isInitialising) {
            this->isInitialising = isInitialising;
        }

    private:

        /** Super class. */
        typedef Call Base;

        /** The intents that are provided by the call. */
        static const char *INTENTS[1];

#ifdef WITH_MPI
        /** The communicator that the caller should use. */
        MPI_Comm comm;
#endif /* WITH_MPI */

        /** The node colour that should be used when initialising MPI. */
        int colour;

        /** Determines whether the initialisation has been performed. */
        bool isInitialising;
    };

    typedef CallAutoDescription<MpiCall> MpiCallDescription;

} /* end namespace mpi */
} /* end namespace cluster */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CLUSTER_MPI_MPICALL_H_INCLUDED */
