/*
 * NetworkInformation.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_NETWORKINFORMATION_H_INCLUDED
#define VISLIB_NETWORKINFORMATION_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include "vislib/IPAddress.h"   // Must be included at begin!
#include "vislib/Socket.h"      // Must be included at begin!
#include "vislib/String.h"
#include "vislib/SmartPtr.h"
#include "vislib/ArrayAllocator.h"


namespace vislib {
namespace net {


    /**
     * Utility class for informations about the local network configuration.
     */
    class NetworkInformation {

    public:

        /**
         * Nested class for adapter informations
         */
        class Adapter {

            /** friend class for initializing the values */
            friend class NetworkInformation;

        public:

            /** possible values for validity of attributes */
            enum ValidityType {
                NOT_VALID,      /* value is not valid. Do not use it. */
                VALID,          /* value is valid. */
                VALID_GENERATED /* a valid value could be generated, which 
                                 * might be inconsistent with the system.
                                 */
            };

            /** Ctor. */
            Adapter(void);

            /** Dtor. */
            ~Adapter(void);

            /**
             * Answer the name of the adapter.
             *
             * @return A reference to the name of the adapter.
             */
            inline const vislib::StringA& Name(void) const {
                return this->name;
            }

            /**
             * Answer the validity of the name.
             *
             * @return The validity of the name.
             */
            inline ValidityType NameValidity(void) const {
                return this->nameValid;
            }

            /**
             * Answer the IP4-Address of the adapter.
             *
             * @return A reference to the address of the adapter.
             */
            inline const IPAddress& Address(void) const {
                return this->address;
            }

            /**
             * Answer the validity of the IP4-Address.
             *
             * @return The validity of the IP4-Address.
             */
            inline ValidityType AddressValidity(void) const {
                return this->addressValid;
            }

            /**
             * Answer the IP4-Subnet-Mask of the adapter.
             *
             * @return A reference to the subnet-mask of the adapter.
             */
            inline const IPAddress& SubnetMask(void) const {
                return this->netmask;
            }

            /**
             * Answer the validity of the IP4-Subnet-Mask.
             *
             * @return The validity of the IP4-Subnet-Mask.
             */
            inline ValidityType SubnetMaskValidity(void) const {
                return this->netmaskValid;
            }

            /**
             * Answer the IP4-Broadcast address of the adapter. If the system
             * does not provide this address directly, it's calculated by the
             * IP4-Address and the IP4-Subnet-Mask.
             *
             * @return A reference to the broadcast address of the adapter.
             */
            inline const IPAddress& BroadcastAddress(void) const {
                return this->broadcast;
            }

            /**
             * Answer the validity of the IP4-Broadcast address.
             *
             * @return The validity of the IP4-Broadcast address.
             */
            inline ValidityType BroadcastAddressValidity(void) const {
                return this->broadcastValid;
            }

            /**
             * Answer the MAC-address of the adapter. The string might be empty
             * if the system cannot access the hardware address.
             *
             * @return A reference to the MAC-address.
             */
            inline const vislib::StringA& MACAddress(void) const {
                return this->mac;
            }

            /**
             * Answer the validity of the MAC-address.
             *
             * @return The validity of the MAC-address.
             */
            inline ValidityType MACAddressValidity(void) const {
                return this->macValid;
            }

        private:

            /** The name of the adapter */
            vislib::StringA name;

            /** The IP4-Address of the adapter */
            IPAddress address;

            /** The IP4-Subnetmask of the adapter */
            IPAddress netmask;

            /** The IP4-Broadcast address of the adapter */
            IPAddress broadcast;

            /** The Hardware MAC address of the adapter */
            vislib::StringA mac;

            /** The validity of the name attribute */
            ValidityType nameValid;

            /** The validity of the address attribute */
            ValidityType addressValid;

            /** The validity of the subnet mask attribute */
            ValidityType netmaskValid;

            /** The validity of the broadcast address attribute */
            ValidityType broadcastValid;

            /** The validity of the mac address attribute */
            ValidityType macValid;

        };

        /**
         * Answer the number of network adapters.
         *
         * @return The number of network adapters.
         */
        static unsigned int AdapterCount(void);

        /**
         * Returns a reference to the information object of the i-th network 
         * adapter.
         *
         * @param i The number of the network adapter.
         *
         * @return A reference to the adapter information object.
         *
         * @throw OutOfRangeException if i is larger or equal the number of
         *        network adapters returned by 'AdapterCount'
         */
        static const Adapter& AdapterInformation(unsigned int i);

    private:

        /**
         * Initializes the list of network adapter objects
         */
        static void InitAdapters(void);

        /** forbidden Ctor. */
        NetworkInformation(void);

        /** forbidden copy Ctor. */
        NetworkInformation(const NetworkInformation& rhs);

        /** forbidden Dtor. */
        ~NetworkInformation(void);

        /** number of network adapters */
        static unsigned int countNetAdapters;

        /** list of network adapters */
        static vislib::SmartPtr<Adapter, vislib::ArrayAllocator<Adapter> > netAdapters;

    };
    
} /* end namespace net */
} /* end namespace vislib */

#endif /* VISLIB_NETWORKINFORMATION_H_INCLUDED */

