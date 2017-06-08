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

#include <atomic>
#include <climits>

#ifdef _WIN32
#include <WinSock2.h>
#include <windows.h>
#include "vislib/graphics/gl/IncludeAllGL.h"
#endif /* _WIN32 */

#ifdef WITH_MPI
#include <mpi.h>
#endif /* WITH_MPI */

#include "mmcore/CallerSlot.h"

#include "mmcore/cluster/simple/View.h"

#include "mmcore/param/ParamSlot.h"
#include "mmcore/param/ParamUpdateListener.h"

#include "vislib/net/AbstractSimpleMessage.h"
#include "vislib/sys/CriticalSection.h"
#include "vislib/RawStorage.h"
#include "vislib/RawStoragePool.h"
#include "vislib/Serialiser.h"
#include "vislib/net/SimpleMessageDispatchListener.h"
#include "vislib/SmartPtr.h"
#include "vislib/String.h"

#define MPI_VIEW_WITH_SWAPGROUP


namespace megamol {
namespace core {
namespace cluster {
namespace mpi {


    /**
     * Abstract base class of override rendering views
     */
    class View : public megamol::core::cluster::simple::View,
            public vislib::net::SimpleMessageDispatchListener {

    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static inline const char *ClassName(void) {
            return "MpiClusterView";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static inline const char *Description(void) {
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

        virtual void OnDispatcherExited(
            vislib::net::SimpleMessageDispatcher& src) throw();

        virtual void OnDispatcherStarted(
            vislib::net::SimpleMessageDispatcher& src) throw();

        virtual bool OnMessageReceived(
            vislib::net::SimpleMessageDispatcher& src,
            const vislib::net::AbstractSimpleMessage& msg) throw();

        /**
         * Renders this AbstractView3D in the currently active OpenGL context.
         *
         * @param context
         */
        virtual void Render(const mmcRenderViewContext& context);

    protected:

        /**
         * Encapsulates NVIDIA Gsync functionality.
         */
        class SwapGroupApi {

        public:

            /**
             * Get the single instance of SwapGroupApi.
             *
             * @return The SwapGroupApi singleton.
             */
            static SwapGroupApi& GetInstance(void);

            bool BindSwapBarrier(const GLuint group, const GLuint barrier);

            bool JoinSwapGroup(const GLuint group);

            /**
             * Answer whether the whole swap group functionality is available.
             *
             * @return true if the swap group APIs can be used, false otherwise.
             */
            inline bool IsAvailable(void) const {
                return this->isAvailable;
            }

            bool QueryFrameCount(unsigned int &count);

            bool QueryMaxSwapGroups(GLuint& outMaxGroups,
                GLuint& outMaxBarriers);

            bool QuerySwapGroup(GLuint& outGroup, GLuint& outBarrier);

            bool ResetFrameCount(void);

        private:

            /**
             * Initialise a new instance.
             */
            SwapGroupApi(void);

            /** Remembers whether ALL function pointers could be acquired. */
            bool isAvailable;

#if (defined(_WIN32) && defined(MPI_VIEW_WITH_SWAPGROUP))
            PFNWGLJOINSWAPGROUPNVPROC wglJoinSwapGroupNV;
            PFNWGLBINDSWAPBARRIERNVPROC wglBindSwapBarrierNV;
            PFNWGLQUERYSWAPGROUPNVPROC wglQuerySwapGroupNV;
            PFNWGLQUERYMAXSWAPGROUPSNVPROC wglQueryMaxSwapGroupsNV;
            PFNWGLQUERYFRAMECOUNTNVPROC wglQueryFrameCountNV;
            PFNWGLRESETFRAMECOUNTNVPROC wglResetFrameCountNV;
#endif /* (defined(_WIN32) && defined(MPI_VIEW_WITH_SWAPGROUP)) */
        };

        /** The status block for each frame. */
        typedef struct FrameState {
            double Time;
            double InstanceTime;
            bool InvalidateMaster;
            bool InitSwapGroup;
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
         * Copies all unique messages from 'relayBuffer' into
         * 'filteredRelayBuffer' and returns the size of the latter.
         *
         * The lock for the relay buffer is acquired by the method while
         * accessing 'relayBuffer'.
         *
         * @return The size of the filtered message.
         */
        size_t filterRelayBuffer(void);

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
            return this->bcastMaster;
        }

        /**
         * Answer whether NVIDIA Gsync is available on this machine.
         */
        bool hasGsync(void) const;

        /**
         * Initialise MPI.
         *
         * If a call is registered for 'callRequestMpi', we use this call to let
         * another module initialise MPI. Otherwise, we do it ourselve. This
         * ensures that existing MPI-based projects can work as they did before
         * the introduction of MpiProvider. If we initialised MPI ourselve, we
         * set 'isMpiInitialised' to true and finalise it once the the view is
         * destroyed.
         *
         * Initialisation of MPI is lazy: If 'comm' is not MPI_COMM_NULL, the
         * method does nothing.
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
            return (this->bcastMaster == this->mpiRank);
        }

        /**
         * Answer whether the machine supports NVIDIA Gsync and whether it is
         * currently enabled.
         *
         * @return true if Gsync is supported and enabled, false otherwise.
         */
        bool isGsyncEnabled(void) const;

        /**
         * Answer whether we know a valid master rank for MPI broadcasts.
         *
         * @return true if the master is known, false otherwise.
         */
        inline bool knowsBcastMaster(void) const {
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
         * Implementation of 'Release'.
         */
        virtual void release(void);

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
        typedef vislib::net::SimpleMessageDispatchListener Base2;

        /**
         * Rank of the broadcast master (this is the node that has a connection
         * to the remote controller node).
         */
        int bcastMaster;

        /* Call to initialise MPI. */
        CallerSlot callRequestMpi;

#ifdef WITH_MPI
        /** The communicator that the view uses. */
        MPI_Comm comm;
#endif /* WITH_MPI */

        /**
         * The buffer that is acutually transmitted. This buffer contains
         * filtered messages only to prevent superseded data from being
         * transferred.
         */
        vislib::RawStorage filteredRelayBuffer;

        /**
         * Remembers whether the registered client has a connection to the 
         * controller node that generates the initial updates.
         */
        std::atomic_bool hasMasterConnection;

        /** Remembers whether MPI was initialised (by the view!). */
        bool isMpiInitialised;

        ///** A memory pool for composing messages etc. */
        //vislib::RawStoragePool memPool;

        /** The rank of this instance in the display communicator. */
        int mpiRank;

        /** The size of the display communicator. */
        int mpiSize;

        /**
         * Remembers whether the master node must be re-negotiated in the next
         * frame.
         */
        bool mustNegotiateMaster;

        /** Configures whether the view should try to enable GSync. */
        param::ParamSlot paramUseGsync;

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
