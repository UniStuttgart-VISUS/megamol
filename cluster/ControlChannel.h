/*
 * ControlChannel.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CONTROLCHANNEL_H_INCLUDED
#define MEGAMOLCORE_CONTROLCHANNEL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "vislib/AbstractBidiCommChannel.h"
#include "vislib/AbstractSimpleMessage.h"
#include "vislib/Listenable.h"
#include "vislib/RunnableThread.h"
#include "vislib/SimpleMessageDispatcher.h"
#include "vislib/SimpleMessageDispatchListener.h"
#include "vislib/SmartRef.h"


namespace megamol {
namespace core {
namespace cluster {

    /**
     * class for control communication channel end points
     */
    class ControlChannel : public vislib::Listenable<ControlChannel>,
        protected vislib::net::SimpleMessageDispatchListener {
    public:

        /**
         * Class for listener object
         */
        class Listener : public vislib::Listenable<ControlChannel>::Listener {
        public:

            /** Ctor */
            Listener(void) {
            }

            /** Dtor */
            virtual ~Listener(void) {
            }

            /**
             * Informs that the control channel is now connected an can send and receive messages
             *
             * @param sender The sending object
             */
            virtual void OnControlChannelConnect(ControlChannel& sender) {
            }

            /**
             * Informs that the control channel is no longer connected.
             *
             * @param sender The sending object
             */
            virtual void OnControlChannelDisconnect(ControlChannel& sender) {
            }

            /**
             * A message has been received over the control channel.
             *
             * @param sender The sending object
             * @param msg The received message
             */
            virtual void OnControlChannelMessage(ControlChannel& sender,
                    const vislib::net::AbstractSimpleMessage& msg) {
            }

        };

        /**
         * Ctor
         */
        ControlChannel(void);

        /**
         * Dtor.
         */
        virtual ~ControlChannel(void);

        /**
         * Closes the communication channel
         */
        void Close(void);

        /**
         * Answer if the channel is open and ready to send or receive
         *
         * @return True if the channel is open.
         */
        bool IsOpen(void) const;

        /**
         * Opens the control channel on the given already opened channel.
         * To clarify: this only sets the channel, takes ownership of the
         * channel object, and starts a receiver thread.
         *
         * @param channel The channel to be used
         */
        void Open(vislib::SmartRef<vislib::net::AbstractBidiCommChannel> channel);

        /**
         * Sends a message to all nodes in the cluster.
         *
         * @param msg The message to be send
         */
        void SendMessage(const vislib::net::AbstractSimpleMessage& msg);

        /**
         * Test for equality
         *
         * @param rhs The right hand side operand
         *
         * @return True if this and rhs are equal
         */
        bool operator==(const ControlChannel& rhs) const;

        /**
         * Assignment operator. Calling this is only valid if the channel is NOT open.
         *
         * @param rhs The right hand side operand
         *
         * @return A reference to this object
         */
        ControlChannel& operator=(const ControlChannel& rhs);

    protected:

        /**
         * This method is called once a communication error occurs.
         *
         * This method should return very quickly and should not perform
         * excessive work as it is executed in the discovery thread.
         *
         * The return value of the method can be used to stop the message
         * dispatcher. The default implementation returns true for continuing
         * after an error.
         *
         * Note that the dispatcher will stop if any of the registered listeners
         * returns false.
         *
         * @param src       The SimpleMessageDispatcher which caught the 
         *                  communication error.
         * @param exception The exception that was caught (this exception
         *                  represents the error that occurred).
         *
         * @return true in order to make the SimpleMessageDispatcher continue
         *         receiving messages, false will cause the dispatcher to
         *         exit.
         */
        virtual bool OnCommunicationError(vislib::net::SimpleMessageDispatcher& src, const vislib::Exception& exception) throw();

        /**
         * This method is called immediately after the message dispatcher loop
         * was left and the dispatching method is being exited.
         *
         * This method should return very quickly and should not perform
         * excessive work as it is executed in the discovery thread.
         *
         * @param src The SimpleMessageDispatcher that exited.
         */
        virtual void OnDispatcherExited(vislib::net::SimpleMessageDispatcher& src) throw();

        /**
         * This method is called immediately before the message dispatcher loop
         * is entered, but after the dispatcher was initialised. This method
         * can be used to release references to the communication channel that
         * the caller has and does not need any more.
         *
         * This method should return very quickly and should not perform
         * excessive work as it is executed in the discovery thread.
         *
         * @param src The SimpleMessageDispatcher that exited.
         */
        virtual void OnDispatcherStarted(vislib::net::SimpleMessageDispatcher& src) throw();


        /**
         * This method is called every time a message is received.
         *
         * This method should return very quickly and should not perform
         * excessive work as it is executed in the discovery thread.
         *
         * The return value of the method can be used to stop the message
         * dispatcher, e. g. if an exit message was received. 
         *
         * Note that the dispatcher will stop if any of the registered listeners
         * returns false.
         *
         * @param src The SimpleMessageDispatcher that received the message.
         * @param msg The message that was received.
         *
         * @return true in order to make the SimpleMessageDispatcher continue
         *         receiving messages, false will cause the dispatcher to
         *         exit.
         */
        virtual bool OnMessageReceived(vislib::net::SimpleMessageDispatcher& src, const vislib::net::AbstractSimpleMessage& msg) throw();

    private:

        /**
         * Forbidden copy ctor
         */
        ControlChannel(const ControlChannel& src);

        /** The communication channel */
        vislib::SmartRef<vislib::net::AbstractBidiCommChannel> channel;

        /** The receiver thread */
        vislib::sys::RunnableThread<vislib::net::SimpleMessageDispatcher> receiver;

    };


} /* end namespace cluster */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CONTROLCHANNEL_H_INCLUDED */
