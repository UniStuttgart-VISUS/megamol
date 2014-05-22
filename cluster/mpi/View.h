/*
 * View.h
 *
 * Copyright (C) 2014 Visualisierungsinstitut der Universität Stuttgart.
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CLUSTER_MPI_VIEW_H_INCLUDED
#define MEGAMOLCORE_CLUSTER_MPI_VIEW_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include <climits>

#include "CallerSlot.h"

#include "cluster/simple/View.h"

#include "param/ParamSlot.h"
#include "param/ParamUpdateListener.h"

#include "vislib/AbstractSimpleMessage.h"
#include "vislib/CriticalSection.h"
#include "vislib/RawStorage.h"
#include "vislib/RawStoragePool.h"
#include "vislib/Serialiser.h"
#include "vislib/SmartPtr.h"
#include "vislib/StackTrace.h"
#include "vislib/String.h"


namespace megamol {
namespace core {
namespace cluster {
namespace mpi {


    /**
     * Abstract base class of override rendering views
     */
    class View : public megamol::core::cluster::simple::View,
            public megamol::core::param::ParamUpdateListener {

    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static inline const char *ClassName(void) {
            VLAUTOSTACKTRACE;
            return "MpiClusterView";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static inline const char *Description(void) {
            VLAUTOSTACKTRACE;
            return "MPI-based powerwall view module.";
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
            VLAUTOSTACKTRACE;
            return false;
        }

        /**
         * Initialises a new instance.
         */
        View(void);

        /**
         * Finalises the instance.
         */
        virtual ~View(void);

        virtual void ConnectView(const vislib::StringA& toName);

        virtual void OnControllerConnectionChanged(const bool isConnected);

        virtual void ParamUpdated(param::ParamSlot& slot);

        /**
         * Renders this AbstractView3D in the currently active OpenGL context.
         *
         * @param time
         * @param instTime
         */
        virtual void Render(float time, double instTime);

    protected:

        /** The status block for each frame. */
        typedef struct FrameState {
            float Time;
            double InstanceTime;
            bool InvalidateMaster;
            size_t RelaySize;
        } FrameState;

        /** Defines the state that the view is in. */
        typedef enum ViewState {
            CREATED,
            LAZY_INITIALISED,       //< Indicates first frame was rendered.
            FORCE_UINT = UINT_MAX
        } ViewState;

        /**
         * Implementation of 'Create'.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        virtual bool create(void);

        /**
         * Finalise MPI, but only if it was initialised by this object.
         */
        virtual void finaliseMpi(void);

        /**
         * Gets the master rank for MPI broadcasts.
         *
         * @return The master rank for broadcasts or a negative value if not
         *         known.
         */
        inline int getBcastMaster(void) const {
            VLAUTOSTACKTRACE;
            return this->bcastMaster;
        }

        /**
         * Initialise MPI if 'paramInitialiseMpi' indicates that the object
         * should do so and the object has not yet initialised MPI.
         *
         * @return true if MPI was successfully initialised or if someone else
         *         should have done this; false if the operation was requested,
         *         but failed.
         */
        virtual bool initialiseMpi(void);

        /**
         * Answer whether this node is the master rank for MPI broadcasts.
         *
         * @return true if this rank is the master, false otherwise.
         */
        inline bool isBcastMaster(void) const {
            VLAUTOSTACKTRACE;
            return (this->bcastMaster == this->mpiRank);
        }

        /**
         * Answer whether we know a valid master rank for MPI broadcasts.
         *
         * @return true if the master is known, false otherwise.
         */
        inline bool knowsBcastMaster(void) const {
            VLAUTOSTACKTRACE;
            return (this->bcastMaster >= 0);
        }

        /**
         * Negotiate the master for MPI broadcasts.
         *
         * This method tries to identify the rank of a node that has a
         * connection to the remote controller machine.
         *
         * @return true if a master node could be negotiated, false otherwise.
         */
        bool negotiateBcastMaster(void);

        /**
         * Handles changes of 'paramInitialiseMpi'.
         *
         * @param slot
         *
         * @return
         */
        virtual bool onInitialiseMpiChanged(param::ParamSlot& slot);

        /**
         * Forward the initialisation message to all nodes and then perform
         * the standard processing.
         *
         * @param msg The initialisation message to be processed.
         */
        virtual void onProcessInitialisationMessage(
            const vislib::net::AbstractSimpleMessage& msg);

        /**
         * Implementation of 'Release'.
         */
        virtual void release(void);

        void storeCameraForRelay(void);

        /**
         * Store the given message for relay to other nodes.
         *
         * This method has no effect if the caller is not the relay master node.
         *
         * @param msg
         */
        void storeMessageForRelay(
            const vislib::net::AbstractSimpleMessage& msg);

    private:

        /** Base class typedef. */
        typedef megamol::core::cluster::simple::View Base1;

        /** Base class typedef. */
        typedef megamol::core::param::ParamUpdateListener Base2;

        /**
         * Rank of the broadcast master (this is the node that has a connection
         * to the remote controller node).
         */
        int bcastMaster;

        /**
         * Remembers whether the registered client has a connection to the 
         * controller node that generates the initial updates.
         */
        bool hasMasterConnection;

        /** Remembers whether MPI was initialised. */
        bool isMpiInitialised;

        /* A memory pool for composing messages etc. */
        vislib::RawStoragePool memPool;

        /** The rank of this instance in MPI_COMM_WORLD. */
        int mpiRank;

        /** The siez of MPI_COMM_WORLD. */
        int mpiSize;

        /** Configures whether this view will initialise MPI. */
        param::ParamSlot paramInitialiseMpi;

        /**
         * The buffer used to compose the status that is relayed to all 
         * nodes before rendering the next frame.
         */
        vislib::RawStorage relayBuffer;

        /** Lock for protecting 'relayBuffer'. */
        vislib::sys::CriticalSection relayBufferLock;

        /** The offset of the next part to be added to 'relayBuffer'. */
        size_t relayOffset;

        /** Remembers the state of this view. */
        ViewState viewState;

    };

} /* end namespace mpi */
} /* end namespace cluster */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CLUSTER_MPI_VIEW_H_INCLUDED */
