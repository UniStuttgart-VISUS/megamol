/*
* AbstractSimpleMessage.h
*
* Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart. 
* Alle Rechte vorbehalten.
*/

#ifndef VISLIB_ABSTRACTSIMPLEMESSAGE_H_INCLUDED
#define VISLIB_ABSTRACTSIMPLEMESSAGE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/SimpleMessageHeader.h"
#include "vislib/ShallowSimpleMessageHeader.h"


namespace vislib {
    namespace net {


        /**
        * This class defines the interface of a network message for use with the 
        * vislib-defined simple protocol. There are two child classes, which 
        * implement a message that manages its own memory and one that uses 
        * user-specified memory. Please note that operations that reallocate the
        * message storage might fail if user-given memory is used.
        *
        * Implementation note: For all implementations, it is assumed that the
        * message header and the message body form a continous memory range.
        */
        class AbstractSimpleMessage {

        public:

            /** Dtor. */
            virtual ~AbstractSimpleMessage(void);

            /**
             * Ensure that the message body is big enough to hold the number of
             * bytes specified in the message header.
             *
            * @throws Exception or derived in case of an error. 
             */
            void AssertBodySize(void) {
                VLSTACKTRACE("AbstractSimpleMessage::AssertBodySize", 
                    __FILE__, __LINE__);
                this->assertStorage(this->GetHeader().GetBodySize());
            }

            /**
            * Get a pointer to the message body. The object remains owner of
            * the memory designated by the pointer returned.
            *
            * @return A pointer to the message body.
            */
            const void *GetBody(void) const;

            /**
            * Get a pointer to the message body. The object remains owner of
            * the memory designated by the pointer returned.
            *
            * @return A pointer to the message body.
            */
            void *GetBody(void);

            /**
            * Answer the message body as pointer to T. The object remains
            * owner of the memory designated by the pointer returned.
            *
            * @return The message body.
            */
            template<class T> inline const T *GetBodyAs(void) const {
                VLSTACKTRACE("AbstractSimpleMessage::GetBodyAs", 
                    __FILE__, __LINE__);
                return reinterpret_cast<const T *>(this->GetBody());
            }

            /**
            * Answer the message body as pointer to T. The object remains
            * owner of the memory designated by the pointer returned.
            *
            * @return The message body.
            */
            template<class T> inline T *GetBodyAs(void) {
                VLSTACKTRACE("AbstractSimpleMessage::GetBodyAs", 
                    __FILE__, __LINE__);
                return reinterpret_cast<T *>(this->GetBody());
            }

            /**
            * Answer the message body at an offset of 'offset' bytes as pointer 
            * to T. The object remains owner of the memory designated by the 
            * pointer returned.
            *
            * Note that the method does not perform any range checks!
            *
            * @return A pointer to the begin of the message body + 'offset' bytes.
            */
            template<class T> 
            inline const T *GetBodyAsAt(const SIZE_T offset) const {
                VLSTACKTRACE("AbstractSimpleMessage::GetBodyAsAt", 
                    __FILE__, __LINE__);
                return reinterpret_cast<const T *>(this->GetBodyAs<BYTE>() 
                    + offset);
            }

            /**
            * Answer the message body at an offset of 'offset' bytes as pointer 
            * to T. The object remains owner of the memory designated by the 
            * pointer returned.
            *
            * Note that the method does not perform any range checks!
            *
            * @return A pointer to the begin of the message body + 'offset' bytes.
            */
            template<class T> 
            inline T *GetBodyAsAt(const SIZE_T offset) {
                VLSTACKTRACE("AbstractSimpleMessage::GetBodyAsAt", 
                    __FILE__, __LINE__);
                return reinterpret_cast<T *>(this->GetBodyAs<BYTE>() + offset);
            }

            /**
            * Answer the message header. Please note, that the returned object is
            * an alias, i. e. all changes to the header directly affect the 
            * message.
            */
            inline const ShallowSimpleMessageHeader& GetHeader(void) const {
                VLSTACKTRACE("AbstractSimpleMessage::GetHeader", 
                    __FILE__, __LINE__);
                return this->header;
            }

            /**
            * Answer the message header. Please note, that the returned object is
            * an alias, i. e. all changes to the header directly affect the 
            * message.
            */
            inline ShallowSimpleMessageHeader& GetHeader(void) {
                VLSTACKTRACE("AbstractSimpleMessage::GetHeader", 
                    __FILE__, __LINE__);
                return this->header;
            }

            /**
            * Answer the overall message size in bytes.
            *
            * @return The combined size of the header and message body.
            */
            inline SIZE_T GetMessageSize(void) const {
                VLSTACKTRACE("AbstractSimpleMessage::GetMessageSize", 
                    __FILE__, __LINE__);
                return (this->GetHeader().GetHeaderSize() 
                    + this->GetHeader().GetBodySize());
            }

            /**
            * Update the message body with data from 'body'.
            *
            * Please note that for ShallowSimpleMessages, passing a body that is
            * too large to fit the existing memory block fill fail with an 
            * exception.
            *
            * @param body     Pointer to the body data of at least 'bodySize' or
            *                 this->GetHeader().GetBodySize() bytes, depending on
            *                 the value of 'bodySize'. If this pointer is NULL, 
            *                 the message body will be erased regardless of the
            *                 value of 'bodySize'. The caller remains owner of
            *                 the memory designated by 'body'. The object creates
            *                 a deep copy in this method.
            * @param bodySize The size of the data to be copied to the message 
            *                 body in bytes. If this parameter is 0, the value
            *                 from the current message header is used. If 'body'
            *                 is NULL, this parameter is ignored and the new 
            *                 body size is 0.
            *
            * @throws Exception or derived in case of an error.
            */
            void SetBody(const void *body, const SIZE_T bodySize = 0);

            /**
            * Update the message header. 
            *
            * Please note that for ShallowSimpleMessages, it is not recommended to
            * reallocate the message as this will fail if the reallocation requires
            * growing the storage.
            *
            * Please note, that the body might be erased if 'reallocateBody' is
            * true, which is the default.
            *
            * @param header         The new message header. The object will create 
            *                       a deep copy anyway.
            * @param reallocateBody If true, the message storage will be 
            *                       reallocated to fit the new body size.
            *
            * @throws Exception or derived in case of an error.
            */
            void SetHeader(const AbstractSimpleMessageHeader& header, 
                const bool reallocateBody = true);

            /**
            * Assignment operator.
            *
            * Note for subclass implementors: This implementation allocates 
            * sufficient memory for 'rhs' in this message using assertStorage
            * and then memcopies the whole message. A check of the storage pointers
            * is done before copying, i. e. the method will not copy from an
            * aliased object (if this is recognisable).
            *
            * @param rhs The right hand side operand.
            *
            * @return *this.
            */
            AbstractSimpleMessage& operator =(const AbstractSimpleMessage& rhs);

            /**
            * Test for equality.
            *
            * @param rhs The right hand side operand.
            *
            * @return true if this message and 'rhs' are equal (header and body),
            *         false otherwise.
            */
            bool operator ==(const AbstractSimpleMessage& rhs) const;

            /**
            * Test for inequality.
            *
            * @param rhs The right hand side operand.
            *
            * @return true if this message and 'rhs' are not equal (header and 
            *         body),
            *         false otherwise.
            */
            bool operator !=(const AbstractSimpleMessage& rhs) const {
                VLSTACKTRACE("AbstractSimpleMessage::operator !=", 
                    __FILE__, __LINE__);
                return !(*this == rhs);
            }

            /**
             * Get the message data starting at the header. This is what you want to
             * send over the network.
             *
             * The ownership of the memory returned does not change.
             *
             * @return Pointer to the memory of the message (including header).
             */
            operator const void *(void) const;

            /**
             * Get the message data starting at the header. This is what you want to
             * send over the network.
             *
             * The ownership of the memory returned does not change.
             *
             * @return Pointer to the memory of the message (including header).
             */
            operator void *(void);

        protected:

            /** Ctor. */
            AbstractSimpleMessage(void);

            /**
            * Ensure that whatever type of storage is used has enough memory to 
            * store a message (including header) with the specified size. The caller
            * must add the size of the header.
            *
            * Child classes must implement this method in order to allow the logic
            * of this parent class to work. The parent class will care for all 
            * intermediate pointers to be updated and for header data to be 
            * restored in case of a reallocation.
            *
            * @param outStorage This variable receives the pointer to the begin of
            *                   the storage.
            * @param size       The size of the memory to be allocated in bytes.
            *
            * @return true if the storage has been reallocated, false if it remains
            *         the same (i. e. the pointer has not been changed).
            *
            * @throws Exception or derived in case of an error.
            */
            virtual bool assertStorage(void *& outStorage, const SIZE_T size) = 0;

            /**
            * Ensure that there is enough storage for the message header and a body
            * of the specified size. If the storage is reallocated, the header is
            * automatically preserved (not updated!) by this method. A pointer to 
            * the header is returned.
            *
            * Implementation note: This method uses 
            * assertStorage(void *& outStorage, const SIZE_T size) to allocate the
            * storage. Additional logic is wrapped around this call.
            *
            * @param bodySize The size of the message body in bytes.
            *
            * @return The pointer to the begin of the storage (i. e. header).
            *
            * @throws Exception or derived in case of an error.
            */
            void *assertStorage(const SIZE_T bodySize);

        private:

            /** 
            * Stores the message header. If the message storage is updated, this must be
            * updated, too.
            */
            ShallowSimpleMessageHeader header;

        };

    } /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTSIMPLEMESSAGE_H_INCLUDED */
