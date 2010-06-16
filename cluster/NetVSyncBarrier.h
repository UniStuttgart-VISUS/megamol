/*
 * NetVSyncBarrier.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_NETVSYNCBARRIER_H_INCLUDED
#define MEGAMOLCORE_NETVSYNCBARRIER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "vislib/AbstractBidiCommChannel.h"
#include "vislib/RawStorage.h"
#include "vislib/SmartRef.h"
#include "vislib/String.h"


namespace megamol {
namespace core {
namespace cluster {


    /**
     * The network v-sync barrier
     */
    class NetVSyncBarrier {
    public:

        /**
         * Ctor.
         */
        NetVSyncBarrier(void);

        /**
         * ~Dtor.
         */
        virtual ~NetVSyncBarrier(void);

        /**
         * Connects to a network v-sync server
         *
         * @param address The remote address of the server
         *
         * @return true on success
         *
         * @throws vislib::Exception on any critical error
         */
        bool Connect(const vislib::StringA& address);

        /**
         * Disconnects from the network v-sync server
         */
        void Disconnect(void);

        /**
         * Crosses the synchronization barrier
         *
         * @param id The id of the barrier to cross
         */
        void Cross(unsigned char id);

        /**
         * Answer the size of the barrier data available
         *
         * @return The size of the barrier data
         */
        inline unsigned int GetDataSize(void) const {
            return this->dataSize;
        }

        /**
         * Answer the barrier data
         *
         * @return Pointer to the barrier data
         */
        inline const unsigned char * GetData(void) const {
            return this->data.As<unsigned char>();
        }

    private:

        /** The communication channel */
        vislib::SmartRef<vislib::net::AbstractBidiCommChannel> channel;

        /** The barrier payload data */
        vislib::RawStorage data;

        /** The size of the valid data stored in the barrier */
        unsigned int dataSize;

    };


} /* end namespace cluster */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_NETVSYNCBARRIER_H_INCLUDED */
