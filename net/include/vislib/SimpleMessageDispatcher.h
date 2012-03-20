/*
 * SimpleMessageDispatcher.h
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_SIMPLEMESSAGEDISPATCHER_H_INCLUDED
#define VISLIB_SIMPLEMESSAGEDISPATCHER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/Socket.h"              // Must be first!
#include "vislib/AbstractCommChannel.h"
#include "vislib/CriticalSection.h"
#include "vislib/Runnable.h"
#include "vislib/SimpleMessage.h"
#include "vislib/SingleLinkedList.h"
#include "vislib/SmartRef.h"
#include "vislib/StackTrace.h"
#include "vislib/TcpCommChannel.h"
#include "vislib/UdpCommChannel.h"


namespace vislib {
namespace net {

    /* Forward declarations. */
    class SimpleMessageDispatchListener;


    /**
     * This class implements a runnable that is receiving VISlib SimpleMessages
     * via an inbound communication channel. It is intended to be run a a 
     * separate that and therefore is derived from Runnable.
     */
    class SimpleMessageDispatcher : public vislib::sys::Runnable {

    public:

        /**
         * This structure contains the  configuration of the message dispatcher 
         * and is passed to the Run() method of the dispatcher. The dispatcher 
         * will copy the data of the configuration to local storage. Therefore 
         * the caller can release its reference to the resources in the 
         * configuration once the Start() method of the dispatcher thread 
         * returns. The dispatcher will not release any references except for 
         * those it added by itself.
         */
        typedef struct Configuration_t {
            inline Configuration_t(void) {}
            inline Configuration_t(SmartRef<AbstractCommClientChannel> channel) 
                : Channel(channel) {}
            inline Configuration_t(SmartRef<TcpCommChannel> channel)
                : Channel(channel.DynamicCast<AbstractCommClientChannel>()) {}
            inline Configuration_t(SmartRef<UdpCommChannel> channel)
                : Channel(channel.DynamicCast<AbstractCommClientChannel>()) {}

            /** Channel to be used by the dispatcher. */
            SmartRef<AbstractCommClientChannel> Channel;
        } Configuration;

        /** Ctor. */
        SimpleMessageDispatcher(void);

        /** Dtor. */
        ~SimpleMessageDispatcher(void);

        /**
         * Add a new SimpleMessageDispatchListener to be informed about events
         * of this dispatcher.
         *
         * The caller remains owner of the memory designated by 'listener' and
         * must ensure that the object exists as long as the listener is 
         * registered.
         *
         * This method is thread-safe.
         *
         * @param listener The listener to register. This must not be NULL.
         */
        void AddListener(SimpleMessageDispatchListener *listener);

        /**
         * Get the communication channel the dispatcher is receiving data from.
         * Callers should never receive from this channel on their own!
         *
         * This method is not thread-safe!
         *
         * @return The channel used for receiving data. DO NOT RECEIVE DATA ON
         *         THIS CHANNEL OR MANIPULATE THE SETTINGS OF THE CHANNEL!
         */
        inline SmartRef<AbstractCommClientChannel> GetChannel(void) {
            return this->configuration.Channel;
        }

        /** 
         * Answer the configuration of the dispatcher. Callers should never
         * manipulate the configuration while the dispatcher is running.
         *
         * This method is not thread-safe!
         *
         * @return The configuration of the dispatcher.
         */
        inline const Configuration& GetConfiguration(void) const {
            return this->configuration;
        } 

        /**
         * Startup callback of the thread. The Thread class will call that 
         * before Run().
         *
         * @param config A pointer to the Configuration, which specifies the
         *               settings of the dispatcher.
         */
        virtual void OnThreadStarting(void *config);

        /**
         * Removes, if registered, 'listener' from the list of objects informed
         * about events events.
         * 
         * The caller remains owner of the memory designated by 'listener'.
         *
         * This method is thread-safe.
         *
         * @param listener The listener to be removed. Nothing happens, if the
         *                 listener was not registered.
         */
        void RemoveListener(SimpleMessageDispatchListener *listener);

        /**
         * Perform the work of a thread.
         *
         * @param config A pointer to the Configuration, which specifies the
         *               settings of the dispatcher.
         *
         * @return The application dependent return code of the thread. This 
         *         must not be STILL_ACTIVE (259).
         */
        virtual DWORD Run(void *config);

        /**
         * Abort the work of the dispatcher by forcefully closing the underlying
         * communication channel.
         *
         * @return true.
         */
        virtual bool Terminate(void);

    private:

        /** A thread-safe list for the message listeners. */
        typedef SingleLinkedList<SimpleMessageDispatchListener *,
            vislib::sys::CriticalSection> ListenerList;

        /**
         * Inform all registered listener about an exception that was caught.
         *
         * This method is thread-safe.
         *
         * @param exception The exception that was caught.
         *
         * @return Accumulated (ANDed) return values of all listeners.
         */
        bool fireCommunicationError(const vislib::Exception& exception);

        /**
         * Inform all registered listener about that the listener is exiting.
         *
         * This method is thread-safe.
         */
        void fireDispatcherExited(void);

        /**
         * Inform all registered listener about that the listener is starting.
         *
         * This method is thread-safe.
         */
        void fireDispatcherStarted(void);

        /**
         * Inform all registered listener about a message that was received.
         *
         * This method is thread-safe.
         *
         * @param msg The message that was received.
         *
         * @return Accumulated (ANDed) return values of all listeners.
         */
        bool fireMessageReceived(const AbstractSimpleMessage& msg);

        /** The configuration of the dispatcher. */
        Configuration configuration;

        /** The list of listeners. */
        ListenerList listeners;

        /** 
         * This object manages the memory of messages that have been received.
         */
        SimpleMessage msg;

    };
    
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_SIMPLEMESSAGEDISPATCHER_H_INCLUDED */

