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


#include "AbstractSimpleMessageHeader.h"


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
        virtual bool OnCommunicationError(SimpleMessageDispatcher& src,
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
        virtual void OnDispatcherExited(SimpleMessageDispatcher& src) throw();

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
        virtual void OnDispatcherStarted(SimpleMessageDispatcher& src) throw();

        ///**
        // * TODO: Documentation
        // */
        //virtual bool OnMessageBodyReceived(
        //    const AbstractSimpleMessageHeader& header,
        //    const void *body) throw();

        ///**
        // * TODO: Documentation
        // * 
        // *
        // * |------------------------------------------------------------|
        // * | Message Header                                             |
        // * |------------------------------------------------------------|
        // * |                                                            |
        // * | A: 'outOffset' bytes                                       |
        // * |                                                            |
        // * |------------------------------------------------------------|
        // * |                                                            |
        // * |                                                            |
        // * | B: 'outDstSize' bytes                                      |
        // * |                                                            |
        // * |                                                            |
        // * |------------------------------------------------------------|
        // *
        // * The dispatcher will call OnMessageReceived and dispatch body part A. 
        // * The message header will be updated to reflect the body size actually
        // * contained in the message part delivered, i. e. 'offset' bytes.
        // * Afterwards the dispatcher will call OnMessageBodyReceived and 
        // * dispatch body part B using the buffer 'outDst'. The message header 
        // * will be updated to reflect the size of the single-copy buffer, i. e. 
        // * 'dstSize' bytes.
        // *
        // * |------------------------------------------------------------|
        // * | Message Header                                             |
        // * |------------------------------------------------------------|
        // * |                                                            |
        // * |                                                            |
        // * | A: 'outDstSize' bytes                                      |
        // * |                                                            |
        // * |                                                            |
        // * |------------------------------------------------------------|
        // * | B: header.GetBodySize() - 'outDstSize' bytes               |
        // * |                                                            |
        // * |------------------------------------------------------------|
        // *
        // * The dispatcher will call OnMessageBodyReceived to deliver body 
        // * part A. The message header will be updated to reflect the body size
        // * actually delivered, i. e. 'dstSize' bytes. Afterwards the dispatcher
        // * will call OnMessageReceived to deliver body parts B. The message 
        // * header will be updated in order to reflect the actual size of the
        // * message part.
        // *
        // * |------------------------------------------------------------|
        // * | Message Header                                             |
        // * |------------------------------------------------------------|
        // * |                                                            |
        // * | A: 'outOffset' bytes                                       |
        // * |                                                            |
        // * |------------------------------------------------------------|
        // * |                                                            |
        // * |                                                            |
        // * | B: 'outDstSize' bytes                                      |
        // * |                                                            |
        // * |                                                            |
        // * |------------------------------------------------------------|
        // * | C: header.GetBodySize() - 'outOffset' - 'outDstSize' bytes |
        // * |                                                            |
        // * |------------------------------------------------------------|
        // * 
        // * This situation occurs if the 'outDst' buffer is too small to receive 
        // * the whole message after 'outOffset'.
        // TODO
        // * 
        // * @param outDst     This variable receives a pointer to the single-copy
        // *                   buffer to which the received message body should be
        // *                   copied. Returning NULL here disables the 
        // *                   single-copy receive operation and will cause the 
        // *                   message to be delivered via OnMessageReceived as a 
        // *                   whole.
        // * @param outDstSize The size of the memory block returned in 'outDst'. A
        // *                   value of 0 for this parameter has the same effect as
        // *                   returning NULL for 'outDst'.
        // * @param outOffset  An offset in bytes for body data which should be 
        // *                   delivered via the standard OnMessageReceived
        // *                   mechanism. Only data after this offset will be 
        // *                   copied to 'outDst'.
        // *                   This parameter has no effect if the single-copy 
        // *                   delivery has been disabled via the value of 
        // *                   'outDst' or 'outDstSize'.
        // * @header           The header that has been received. This header will
        // *                   also be contained in the message delivered to 
        // *                   OnMessageReceived and/or OnMessageBodyReceived. The
        // *                   body size, however, might vary (read above).
        // */
        //virtual void OnMessageHeaderReceived(void *& outDst, 
        //    SimpleMessageSize& outDstSize,
        //    SimpleMessageSize& outOffset,
        //    const AbstractSimpleMessageHeader& header) throw();

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
        virtual bool OnMessageReceived(SimpleMessageDispatcher& src,
            const AbstractSimpleMessage& msg) throw() = 0;

    };
    
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_SIMPLEMESSAGEDISPATCHLISTENER_H_INCLUDED */
