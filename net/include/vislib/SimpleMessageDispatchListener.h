/*
 * SimpleMessageDispatchListener.h
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_SIMPLEMESSAGEDISPATCHLISTENER_H_INCLUDED
#define VISLIB_SIMPLEMESSAGEDISPATCHLISTENER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


    /* Forward declarations. */
namespace vislib {
    class Exception;

namespace net {
    class AbstractInboundCommChannel;
    class AbstractSimpleMessage;
    class SimpleMessageDispatcher;
} /* end namespace net */
} /* end namespace vislib */

namespace vislib {
namespace net {


    /**
     * This class defines the interface for classes that want to be informed
     * about events of a SimpleMessageDispatcher.
     */
    class SimpleMessageDispatchListener {

    public:

        /** Ctor. */
        SimpleMessageDispatchListener(void);

        /** Dtor. */
        virtual ~SimpleMessageDispatchListener(void);

        /**
         * This method is called once a communication error occurs.
         *
         * This method should return very quickly and should not perform
         * excessive work as it is executed in the discovery thread.
         *
         * @param src       The SimpleMessageDispatcher which caught the 
         *                  communication error.
         * @param exception The exception that was caught (this exception
         *                  represents the error that occurred).
         */
        virtual void OnCommunicationError(const SimpleMessageDispatcher& src,
            const vislib::Exception& exception) throw();

        /**
         * This method is called immediately after the message dispatcher loop
         * was left and the dispatching method is being exited.
         *
         * This method should return very quickly and should not perform
         * excessive work as it is executed in the discovery thread.
         *
         * @param src The SimpleMessageDispatcher that exited.
         */
        virtual void OnDispatcherExited(
            const SimpleMessageDispatcher& src) throw();

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
        virtual void OnDispatcherStarted(
            const SimpleMessageDispatcher& src) throw();


        /**
         * This method is called every time a message is received.
         *
         * This method should return very quickly and should not perform
         * excessive work as it is executed in the discovery thread.
         *
         * The return value of the method can be used to stop the message
         * dispatcher, e. g. if an exit message was received. 
         *
         * @param src The SimpleMessageDispatcher that received the message.
         * @param msg The message that was received.
         *
         * @return true in order to make the SimpleMessageDispatcher continue
         *         receiving messages, false will cause the dispatcher to
         *         exit.
         */
        virtual bool OnMessageReceived(const SimpleMessageDispatcher& src,
            const AbstractSimpleMessage& msg) throw() = 0;

    };
    
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_SIMPLEMESSAGEDISPATCHLISTENER_H_INCLUDED */
