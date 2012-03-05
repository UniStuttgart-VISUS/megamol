/*
 * Heartbeat.h
 *
 * Copyright (C) 2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CLUSTER_SIMPLE_HEARTBEAT_H_INCLUDED
#define MEGAMOLCORE_CLUSTER_SIMPLE_HEARTBEAT_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "job/AbstractThreadedJob.h"
#include "Module.h"
#include "CallerSlot.h"
#include "param/ParamSlot.h"
#include "vislib/CommServer.h"
#include "vislib/CommServerListener.h"
#include "vislib/CriticalSection.h"
#include "vislib/Event.h"
#include "vislib/RawStorage.h"
#include "vislib/RunnableThread.h"
#include "vislib/SingleLinkedList.h"
#include "vislib/Thread.h"

//#include "view/AbstractTileView.h"
//#include "vislib/AbstractSimpleMessage.h"
//#include "vislib/Serialiser.h"
//#include "vislib/String.h"


namespace megamol {
namespace core {
namespace cluster {
namespace simple {


    /** forward declaration */
    class Client;


    /**
     * Abstract base class of override rendering views
     */
    class Heartbeat : public job::AbstractThreadedJob, public Module, public vislib::net::CommServerListener {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "SimpleClusterHeartbeat";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Simple cluster module providing a display heartbeat for content lock";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return true;
        }

        /**
         * Disallow usage in quickstarts
         *
         * @return false
         */
        static bool SupportQuickstart(void) {
            return false;
        }

        /** Ctor. */
        Heartbeat(void);

        /** Dtor. */
        virtual ~Heartbeat(void);

        /**
         * Terminates the job thread.
         *
         * @return true to acknowledge that the job will finish as soon
         *         as possible, false if termination is not possible.
         */
        virtual bool Terminate(void);

        /**
         * Unregisters from the specified client
         *
         * @param client The client to unregister from
         */
        void Unregister(class Client *client);

        /**
         * Sets incoming timeCamera data
         *
         * @param data The incoming data
         * @param size The size of the incoming data in bytes
         */
        void SetTCData(const void *data, SIZE_T size);

        /**
         * This method is called once a network error occurs.
         *
         * This method should return very quickly and should not perform
         * excessive work as it is executed in the server thread.
         *
         * @param src       The CommServer which caught the communication error.
         * @param exception The exception that was caught (this exception
         *                  represents the error that occurred).
         *
         * @return true in order to make the CommServer continue listening, 
         *         false will cause the server to exit.
         */
        virtual bool OnServerError(const vislib::net::CommServer& src,
            const vislib::Exception& exception) throw();

        /**
         * The server will call this method when a new client connected. The
         * listener can decide whether it wants to take ownership of the
         * communication channel 'channel' by returning true. If no listener 
         * accepts the new connection, the server will terminate the new 
         * connection by closing it.
         *
         * @param src     The server that made the new channel.
         * @param channel The new communication channel.
         * 
         * @return true if the listener takes ownership of 'channel'. The 
         *         server will not use the channel again. If the method 
         *         returns false, the listener should not use the socket, 
         *         because the server remains its owner.
         */
        virtual bool OnNewConnection(const vislib::net::CommServer& src,
            vislib::SmartRef<vislib::net::AbstractCommClientChannel> channel) throw();

        /**
         * The server will call this method when it left the server loop and
         * is about to exit.
         *
         * @param serv The server that exited.
         */
        virtual void OnServerExited(const vislib::net::CommServer& src) throw();

        /**
         * The server will call this method immediately before entering the 
         * server loop, but after the communication channel was put into
         * listening state.
         *
         * @param serv The server that started.
         */
        virtual void OnServerStarted(const vislib::net::CommServer& src) throw();

    protected:

        /**
         * Implementation of 'Create'.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        virtual bool create(void);

        /**
         * Implementation of 'Release'.
         */
        virtual void release(void);

    private:

        /**
         * An established heartbeat connection
         */
        class Connection {
        public:

            /**
             * Ctor
             *
             * @param parent The owning heartbeat server
             * @param chan The communication channel
             */
            Connection(Heartbeat& parent, vislib::SmartRef<vislib::net::AbstractCommClientChannel> chan);

            /**
             * Closes the connection
             * It is safe to close a closed connection.
             */
            void Close();

            /**
             * Continues
             */
            inline void Continue(void) {
                this->wait.Set();
            }

            /**
             * Gets the data to be sent to the client
             *
             * @return The data to be sent
             */
            inline vislib::RawStorage& Data(void) {
                return this->data;
            }

            /**
             * Gets the flag if this thread is waiting
             *
             * @return True if this thread is waiting
             */
            inline bool IsWaiting(void) const {
                return this->waiting;
            }

        private:

            /**
             * The receiver thread
             *
             * @param userData Points to this object
             *
             * return 0
             */
            static DWORD receive(void *userData);

            /** The owning heartbeat server */
            Heartbeat& parent;

            /** The communication channel */
            vislib::SmartRef<vislib::net::AbstractCommClientChannel> chan;

            /** Flag whether or not this thread is waiting */
            bool waiting;

            /** The communication thread */
            vislib::sys::Thread kun;

            /** The waiting event */
            vislib::sys::Event wait;

            /** The outgoing data */
            vislib::RawStorage data;

        };

        /** Connections may manipulate the list of connections */
        friend class Connection;

        /** The timeCamera buffer type */
        typedef struct _tc_buffer_t {

            /** The instance time */
            double instTime;

            /** The time */
            float time;

            /** The camera setting */
            vislib::RawStorage camera;

            /** The lock for synchronisation */
            vislib::sys::CriticalSection lock;

            /** Flag whether or not the buffer data is valid */
            bool isValid;

        } TCBuffer;

        /**
         * Perform the work of a thread.
         *
         * @param userData A pointer to user data that are passed to the thread,
         *                 if it started.
         *
         * @return The application dependent return code of the thread. This 
         *         must not be STILL_ACTIVE (259).
         */
        virtual DWORD Run(void *userData);

        /**
         * Adds a connection to the list of open connections
         *
         * @param con The connection opened
         */
        void addConn(vislib::SmartPtr<Connection> con);

        /**
         * Removes a connection from the list of open connections
         *
         * @param con The connection to be removed
         */
        void removeConn(Connection *con);

        /**
         * Signals that a connection is now waiting
         *
         * @param con The connection now waiting
         */
        void connWaiting(Connection *con);

        /** The slot registering this view */
        CallerSlot registerSlot;

        /** The client end */
        class Client *client;

        /** Flag letting the thread run */
        bool run;

        /** The lock for the main thread */
        vislib::sys::Event mainlock;

        /** The port of the heartbeat server */
        param::ParamSlot heartBeatPortSlot;

        /** The timeCamera double buffer */
        TCBuffer tcBuf[2];

        /** The index of the current timeCamera buffer */
        unsigned int tcBufIdx;

        /** The heartbeat server */
        vislib::sys::RunnableThread<vislib::net::CommServer> server;

        /** The lock for the connections data */
        vislib::sys::CriticalSection connLock;

        /** The connections */
        vislib::SingleLinkedList<vislib::SmartPtr<Connection> > connList;

        /** The current synchronization tier */
        unsigned char tier;


        ///**
        // * Renders this AbstractView3D in the currently active OpenGL context.
        // */
        //virtual void Render(float time, double instTime);

        ///**
        // * Unregisters from the specified client
        // *
        // * @param client The client to unregister from
        // */
        //void Unregister(class SimpleClusterClient *client);

        ///**
        // * Disconnect the view call
        // */
        //void DisconnectViewCall(void);

        ///**
        // * Set the module graph setup message
        // *
        // * @return msg The message
        // */
        //void SetSetupMessage(const vislib::net::AbstractSimpleMessage& msg);

        ///**
        // * Sets a initialization message for the camera parameters
        // */
        //void SetCamIniMessage(void);

        ///**
        // * Connects this view to another view
        // *
        // * @param toName The slot to connect to
        // */
        //void ConnectView(const vislib::StringA toName);

        ///**
        // * Answer the connected view
        // *
        // * @return The connected view or NULL if no view is connected
        // */
        //inline view::AbstractView *GetConnectedView(void) const {
        //    return this->getConnectedView();
        //}

        ///**
        // * Implementation of 'Create'.
        // *
        // * @return 'true' on success, 'false' otherwise.
        // */
        //virtual bool create(void);

        ///**
        // * Implementation of 'Release'.
        // */
        //virtual void release(void);

        ///**
        // * Renders a fallback view holding information about the cluster
        // */
        //void renderFallbackView(void);

        ///**
        // * Freezes, updates, or unfreezes the view onto the scene (not the
        // * rendering, but camera settings, timing, etc).
        // *
        // * @param freeze true means freeze or update freezed settings,
        // *               false means unfreeze
        // */
        //virtual void UpdateFreeze(bool freeze);

        ///**
        // * Loads the configuration
        // *
        // * @param name The name to load the configuration for
        // *
        // * @return True on success
        // */
        //bool loadConfiguration(const vislib::StringA& name);

        ///**
        // * Event callback when the value of 'directCamSyncSlot' changes
        // *
        // * @param slot directCamSyncSlot
        // *
        // * @return true
        // */
        //bool directCamSyncUpdated(param::ParamSlot& slot);

        ///** Flag to identify the first frame */
        //bool firstFrame;

        ///** Flag if everything is frozen */
        //bool frozen;

        ///** The frozen time */
        //double frozenTime;

        ///** frozen camera parameters */
        //vislib::Serialiser *frozenCam;

        ///** The initialization message */
        //vislib::net::AbstractSimpleMessage *initMsg;

        ///** The address of the heartbeat server */
        //param::ParamSlot heartBeatServerSlot;

        ///** Flag controlling whether or not this view directly syncs it's camera without using the heartbeat server */
        //param::ParamSlot directCamSyncSlot;

    };


} /* end namespace simple */
} /* end namespace cluster */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CLUSTER_SIMPLE_HEARTBEAT_H_INCLUDED */
