/*
 * CommChannel.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_COMMCHANNEL_H_INCLUDED
#define MEGAMOLCORE_COMMCHANNEL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "vislib/Listenable.h"
#include "vislib/SmartRef.h"
#include "vislib/String.h"
#include "vislib/macro_utils.h"
#include "vislib/net/AbstractCommChannel.h"
#include "vislib/net/AbstractSimpleMessage.h"
#include "vislib/net/SimpleMessageDispatchListener.h"
#include "vislib/net/SimpleMessageDispatcher.h"
#include "vislib/sys/RunnableThread.h"


namespace megamol::core::cluster {

/**
 * class for communication channel end points
 */
VISLIB_MSVC_SUPPRESS_WARNING(4251 4275)
class CommChannel : public vislib::Listenable<CommChannel>, protected vislib::net::SimpleMessageDispatchListener {
public:
    /**
     * Class for listener object
     */
    class Listener : public vislib::Listenable<CommChannel>::Listener {
    public:
        /** Ctor */
        Listener() {}

        /** Dtor */
        ~Listener() override {}

        /**
         * Informs that the control channel is now connected an can send and receive messages
         *
         * @param sender The sending object
         */
        virtual void OnCommChannelConnect(CommChannel& sender) {}

        /**
         * Informs that the control channel is no longer connected.
         *
         * @param sender The sending object
         */
        virtual void OnCommChannelDisconnect(CommChannel& sender) {}

        /**
         * A message has been received over the control channel.
         *
         * @param sender The sending object
         * @param msg The received message
         */
        virtual void OnCommChannelMessage(CommChannel& sender, const vislib::net::AbstractSimpleMessage& msg) = 0;
    };

    /**
     * Ctor
     */
    CommChannel();

    /**
     * Copy ctor
     * Using this ctor is only legal if 'src' has no members set to any
     * non-default values.
     *
     * @param src The object to clone from.
     */
    CommChannel(const CommChannel& src);

    /**
     * Dtor.
     */
    ~CommChannel() override;

    /**
     * Closes the communication channel
     */
    void Close();

    /**
     * Answer the counterparts name
     *
     * @return The counterparts name
     */
    inline const vislib::StringA& CounterpartName() const {
        return this->counterpartName;
    }

    /**
     * Answer if the channel is open and ready to send or receive
     *
     * @return True if the channel is open.
     */
    bool IsOpen() const;

    /**
     * Opens the control channel on the given already opened channel.
     * To clarify: this only sets the channel, takes ownership of the
     * channel object, and starts a receiver thread.
     *
     * @param channel The channel to be used
     */
    void Open(vislib::SmartRef<vislib::net::AbstractCommClientChannel> channel);

    /**
     * Sends a message to all nodes in the cluster.
     *
     * @param msg The message to be send
     */
    void SendMessage(const vislib::net::AbstractSimpleMessage& msg);

    /**
     * Sets the name of the connection counterpart
     *
     * @param name The name of the connection counterpart
     */
    void SetCounterpartName(const char* name) {
        this->counterpartName = name;
    }

    /**
     * Test for equality
     *
     * @param rhs The right hand side operand
     *
     * @return True if this and rhs are equal
     */
    bool operator==(const CommChannel& rhs) const;

    /**
     * Assignment operator. Calling this is only valid if the channel is NOT open.
     *
     * @param rhs The right hand side operand
     *
     * @return A reference to this object
     */
    CommChannel& operator=(const CommChannel& rhs);

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
    bool OnCommunicationError(
        vislib::net::SimpleMessageDispatcher& src, const vislib::Exception& exception) throw() override;

    /**
     * This method is called immediately after the message dispatcher loop
     * was left and the dispatching method is being exited.
     *
     * This method should return very quickly and should not perform
     * excessive work as it is executed in the discovery thread.
     *
     * @param src The SimpleMessageDispatcher that exited.
     */
    void OnDispatcherExited(vislib::net::SimpleMessageDispatcher& src) throw() override;

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
    void OnDispatcherStarted(vislib::net::SimpleMessageDispatcher& src) throw() override;


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
    bool OnMessageReceived(
        vislib::net::SimpleMessageDispatcher& src, const vislib::net::AbstractSimpleMessage& msg) throw() override;

private:
    /** The communication channel */
    VISLIB_MSVC_SUPPRESS_WARNING(4251)
    vislib::SmartRef<vislib::net::AbstractCommClientChannel> channel;

    /** The receiver thread */
    VISLIB_MSVC_SUPPRESS_WARNING(4251)
    vislib::sys::RunnableThread<vislib::net::SimpleMessageDispatcher> receiver;

    /** The name of the connection counterpart */
    VISLIB_MSVC_SUPPRESS_WARNING(4251)
    vislib::StringA counterpartName;
};


} // namespace megamol::core::cluster

#endif /* MEGAMOLCORE_COMMCHANNEL_H_INCLUDED */
