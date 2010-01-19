/*
 * RenderNetMsg.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_RENDERNETMSG_H_INCLUDED
#define MEGAMOLCORE_RENDERNETMSG_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "vislib/RawStorage.h"
#include "vislib/types.h"


namespace megamol {
namespace core {
namespace special {

    /**
     * Class storing a network message used by the MegaMol™ rendering network
     */
    class RenderNetMsg {
    public:
        /** Utility class will send and receive the messages */
        friend class RenderNetUtil;

    private:
        
        /**
         * The message header size to be received before the data size member
         * is valid.
         */
        static const unsigned int headerSize = 16;

    public:

        /**
         * Ctor. Creates an empty message with 'type == 0', 'id == 0', and
         * 'size == 0'.
         */
        RenderNetMsg(void);

        /**
         * Ctor.
         *
         * @param type The type of the message
         * @param id The id of the message
         * @param size The size of the message data in bytes
         * @param data The initial message data or 'NULL' (data member will
         *             not be initialized then)
         */
        RenderNetMsg(UINT32 type, UINT32 id, SIZE_T size, const void *data = NULL);

        /**
         * Copy ctor. Makes a deep copy from 'src'.
         *
         * @param src The object to clone from.
         */
        RenderNetMsg(const RenderNetMsg& src);

        /**
         * Dtor.
         */
        ~RenderNetMsg(void);

        /**
         * Consolidates the message data member
         */
        void Consolidate(void);

        /**
         * Access the data member
         *
         * @return The data member
         */
        inline const void *Data(void) const {
            return this->dat.AsAt<void>(RenderNetMsg::headerSize);
        }

        /**
         * Access the data member
         *
         * @return The data member
         */
        inline void *Data(void) {
            return this->dat.AsAt<void>(RenderNetMsg::headerSize);
        }

        /**
         * Access the data member casted to type T
         *
         * @return The data member casted to type T
         */
        template<class T> inline const T *DataAs(void) const {
            return this->dat.AsAt<T>(RenderNetMsg::headerSize);
        }

        /**
         * Access the data member casted to type T
         *
         * @return The data member casted to type T
         */
        template<class T> inline T *DataAs(void) {
            return this->dat.AsAt<T>(RenderNetMsg::headerSize);
        }

        /**
         * Gets the size of the message data.
         *
         * @return The size of the message data
         */
        inline SIZE_T GetDataSize(void) const {
            return static_cast<SIZE_T>(*this->dat.AsAt<UINT64>(8));
        }

        /**
         * Gets the id of the message.
         *
         * @return The id of the message
         */
        inline UINT32 GetID(void) const {
            return *this->dat.AsAt<UINT32>(4);
        }

        /**
         * Gets the type of the message.
         *
         * @return The type of the message
         */
        inline UINT32 GetType(void) const {
            return *this->dat.As<UINT32>();
        }

        /**
         * Sets the size of the message data member.
         *
         * @param size The new of the message data member in bytes.
         * @param keepData Flag if the data stored should remain
         */
        void SetDataSize(SIZE_T size, bool keepData = false);

        /**
         * Sets the id of the message.
         *
         * @param id The new if of the message
         */
        inline void SetID(UINT32 id) {
            *this->dat.AsAt<UINT32>(4) = id;
        }

        /**
         * Sets the type of the message.
         *
         * @param type The new type of the message
         */
        inline void SetType(UINT32 type) {
            *this->dat.As<UINT32>() = type;
        }

        /**
         * Assignment operator. Performs a deep copy.
         *
         * @param rhs The right hand side operand
         *
         * @return Reference to this object
         */
        RenderNetMsg& operator=(const RenderNetMsg& rhs);

    private:

        /** the header and data of the message */
        vislib::RawStorage dat;

    };

} /* end namespace special */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_RENDERNETMSG_H_INCLUDED */
