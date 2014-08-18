/*
 * ShallowSimpleMessage.h
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_SHALLOWSIMPLEMESSAGE_H_INCLUDED
#define VISLIB_SHALLOWSIMPLEMESSAGE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/AbstractSimpleMessage.h"


namespace vislib {
namespace net {


    /**
     * This class can be used to apply the behaviour of a VISlib simple message
     * on a user-supplied memory range. This means that the class creates alias
     * pointers, but does not take ownership of the memory. The user is 
     * responsible for ensuring that the memory lives as long as the message
     * object and that the memory bounds specified are correct.
     *
     * It is illegal to specify NULL pointers. The behaviour of the class is
     * undefined if you do so.
     *
     * It is illegal to specify a memory block that has not at least a size
     * of sizeof(SimpleMessageHeaderData) bytes. the behaivour of the class
     * is undefined if you do so.
     *
     * The class will never release the user-defined memory.
     */
    class ShallowSimpleMessage : public AbstractSimpleMessage {

    public:

        /** 
         * Create a new message using the storage provided by 'storage'. 
         * 'cntStorage' specifies the amount of memory available starting
         * at 'storage'. If this is 0, it is assumed that there is a valid
         * message header at the start of 'storage' which contains the size
         * of the message. This message size is then used as size of the
         * memory block.
         *
         * Be aware of that it is illegal providing storage that is 
         * insufficient for storing a message header. This will cause an
         * unpredictable behaviour.
         *
         * It is illegal providing a NULL pointer for storage. This will
         * cause an unpredictable behaviour.
         *
         * @param storage    Memory to be used for the message. The caller
         *                   remains owner of this memory and must ensure
         *                   that the memory block lives as long as the
         *                   object.
         * @param cntStorage The size of the memory block provided for
         *                   'storage' in bytes. This must be the actual
         *                   size for at least the header or 0 for making
         *                   the ctor interpret the message data.
         */
        ShallowSimpleMessage(void *storage, const SIZE_T cntStorage = 0);

        /** Dtor. */
        virtual ~ShallowSimpleMessage(void);

        /**
         * Update the storage memory.
         *
         * Note that this will not affect the current storage, i. e. there
         * is no ownership change. The caller is responsible for releasing
         * this memory after the method returns.
         *
         * Be aware of that it is illegal providing storage that is 
         * insufficient for storing a message header. This will cause an
         * unpredictable behaviour.
         *
         * It is illegal providing a NULL pointer for storage. This will
         * cause an unpredictable behaviour.
         *
         * @param storage    Memory to be used for the message. The caller
         *                   remains owner of this memory and must ensure
         *                   that the memory block lives as long as the
         *                   object.
         * @param cntStorage The size of the memory block provided for
         *                   'storage' in bytes. This must be the actual
         *                   size for at least the header or 0 for making
         *                   the ctor interpret the message data.
         */
        void SetStorage(void *storage, const SIZE_T cntStorage = 0);

        /**
         * Assignment operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        inline ShallowSimpleMessage& operator =(
                const AbstractSimpleMessage& rhs) {
            VLSTACKTRACE("ShallowSimpleMessage::operator =",
                __FILE__, __LINE__);
            Super::operator =(rhs);
            return *this;
        }

        /**
         * Assignment operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        inline ShallowSimpleMessage& operator =(
                const ShallowSimpleMessage& rhs) {
            VLSTACKTRACE("ShallowSimpleMessage::operator =", 
                __FILE__, __LINE__);
            Super::operator =(rhs);
            return *this;
        }

    protected:

        /**
        * Ensure that whatever type of storage is used has enough memory to 
        * store a message (including header) with the specified size. The caller
        * must add the size of the header.
        *
        * This implementation does NOTHING! The owner of the external storage
        * used by this object is responsible for the memory requirement.
        *
        * @param outStorage This variable receives the pointer to the begin of
        *                   the storage.
        * @param size       The size of the memory to be allocated in bytes.
        *
        * @return false as it remains the same 
        *         (i. e. the pointer has not been changed).
        *
        * @throws Exception or derived in case of an error.
        */
        virtual bool assertStorage(void *& outStorage, const SIZE_T size);

    private:

        /** Superclass typedef. */
        typedef AbstractSimpleMessage Super;
    
        /** User-specified size of the memory allocated for the message. */
        SIZE_T cntStorage;

        /** User-specified memory for the message. */
        void *storage;

    };
    
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_SHALLOWSIMPLEMESSAGE_H_INCLUDED */

