/*
 * NetworkInformation.h
 *
 * Copyright (C) 2009 by Christoph M¸lller. Alle Rechte vorbehalten.
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_NETWORKINFORMATION_H_INCLUDED
#define VISLIB_NETWORKINFORMATION_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/Socket.h"      // Must be included at begin!
#include "vislib/Array.h"
#include "vislib/CriticalSection.h"
#include "vislib/Exception.h"
#include "vislib/IPHostEntry.h"
#include "vislib/StackTrace.h"
#include "vislib/String.h"

#ifdef _WIN32
#include <iphlpapi.h>
#endif /* _WIN32 */

#ifdef _MSC_VER
#pragma comment(lib, "Iphlpapi")
#endif /* _MSC_VER */


namespace vislib {
namespace net {

    /**
     * Utility class for informations about the local network configuration.
     */
    class NetworkInformation {

    public:

        /**
         * This enumeration expresses the confidence of the NetworkInformation 
         * class in the values returned.
         */
        typedef enum Confidence_t {
            INVALID = 0,    //< Value is invalid. Do not use it.
            GUESSED,        //< Value is generated, but might be invalid.
            VALID           //< Value was retrieved from the system.
        } Confidence;


        /**
         * This exception is thrown if a property of an adapter was retrieved 
         * that is obviously invalid, i. e. has a confidence of INVALID.
         */
        class NoConfidenceException : public vislib::Exception {

        public:

            /**
             * Ctor.
             *
             * @param propName Name of the invalid property.
             * @param file     The file the exception was thrown in.
             * @param line     The line the exception was thrown in.
             */
            NoConfidenceException(const char *propName, const char *file, 
                const int line);

            /**
             * Ctor.
             *
             * @param propName Name of the invalid property.
             * @param file     The file the exception was thrown in.
             * @param line     The line the exception was thrown in.
             */
            NoConfidenceException(const wchar_t *propName, const char *file, 
                const int line);

            /**
             * Create a clone of 'rhs'.
             *
             * @param rhs The object to be cloned.
             */
            NoConfidenceException(const NoConfidenceException& rhs);

            /** Dtor. */
            virtual ~NoConfidenceException(void);

            /**
             * Assignment operator.
             *
             * @param rhs The right hand side operand.
             *
             * @return *this.
             */
            virtual NoConfidenceException& operator =(
                const NoConfidenceException& rhs);

        }; /* end class NoConfidenceException */

    private:

        /**
         * This class groups a member with the information about the 
         * validity confidence.
         */
        template<class T> class AssessedMember {

        public:
            
            /**
             * Create a new instance, which is initialised using the default
             * ctor of T and has a confidence value of INVALID.
             */
            AssessedMember(void);

            /**
             * Create a new instance using the specified initial values.
             *
             * @param value      The value of the member.
             * @param confidence The confidence of the value.
             */
            AssessedMember(const T& value, const Confidence confidence);

            /**
             * Clone 'rhs'.
             *
             * @param rhs The object to be cloned.
             */
            inline AssessedMember(const AssessedMember& rhs) {
                *this = rhs;
            }

            /**
             * Dtor.
             */
            ~AssessedMember(void);

            /**
             * Answer the confidence.
             *
             * @return The confidence.
             */
            inline Confidence GetConfidence(void) const {
                return this->confidence;
            }

            /**
             * Answer the confidence to 'outConfidence'.
             *
             * @param outConfidence Receives the confidence. It is safe to 
             *                      pass a NULL pointer.
             * @param name          The name of the property. This is used
             *                      to parametrise the exception.
             *
             * @throws NoConfidenceException If 'outConfidence' is NULL and 
             *                               the confidence value is 
             *                               INVALID.
             */
            void GetConfidence(Confidence *outConfidence, 
                    const char *name) const;

            /**
             * Access the value of the member.
             *
             * @return The value.
             */
            inline const T& GetValue(void) const {
                return this->value;
            }

            /**
             * Set a new confidence value.
             *
             * @param confidence The new confidence value.
             */
            inline void SetConfidence(const Confidence confidence) {
                this->confidence = confidence;
            }

            /**
             * Set a new value for the member.
             *
             * @param value The new value.
             */
            inline void SetValue(const T& value) {
                this->value = value;
            }

            /**
             * Update the AssessedMember.
             *
             * @param value      The new value.
             * @param confidence The new confidence value. 
             */
            inline void Set(const T& value, const Confidence confidence) {
                this->confidence = confidence;
                this->value = value;
            }

            /** 
             * Assignment.
             *
             * @param rhs The right hand side operand.
             *
             * @return *this.
             */
            AssessedMember& operator =(const AssessedMember& rhs);

            /**
             * Access the value of the member.
             *
             * @return The value.
             */
            inline operator T&(void) {
                return this->value;
            }

            /**
             * Access the value of the member.
             *
             * @return The value.
             */
            inline operator const T&(void) const {
                return this->value;
            }

            /**
             * Test for equality.
             *
             * @param rhs The right hand side operand.
             *
             * @return true if this object and 'rhs' are equal,
             *         false otherwise.
             */
            inline bool operator ==(const AssessedMember& rhs) const {
                return ((this->confidence == rhs.confidence)
                    && (this->value == rhs.value));
            }

            /**
             * Test for inequality.
             *
             * @param rhs The right hand side operand.
             *
             * @return true if this object and 'rhs' are not equal,
             *         false otherwise.
             */
            inline bool operator !=(const AssessedMember& rhs) const {
                return !(*this == rhs);
            }

        private:

            /** The confidence in the validity of 'value'. */
            Confidence confidence;

            /** The actual value to be stored. */
            T value;

        }; /* end class AssessedMember */


    public:

        /**
         * This class stores unicast addresses. For unicast addresses, we need 
         * not only to know the address itself, but also the prefix (netmask)
         * associated with it.
         */
        class UnicastAddressInformation {

        public:

            /** 
             * This enumeration defines the possible sources of an address 
             * prefix. 
             */
            typedef enum PrefixOrigin_t {
                PREFIX_ORIGIN_OTHER = 0,            //< Unknown source.
                PREFIX_ORIGIN_MANUAL,               //< Manually specified.
                PREFIX_ORIGIN_WELL_KNOWN,           //< From well-known source.
                PREFIX_ORIGIN_DHCP,                 //< Provided by DHCP.
                PREFIX_ORIGIN_ROUTER_ADVERTISEMENT  //< Provided through RA.
            } PrefixOrigin;

            /** 
             * This enumeration defines the possible sources of an address 
             * suffix. 
             */
            typedef enum SuffixOrigin_t {
                SUFFIX_ORIGIN_OTHER = 0,            //< Unknown source.
                SUFFIX_ORIGIN_MANUAL,               //< Manually specified.
                SUFFIX_ORIGIN_WELL_KNOWN,           //< From well-known source.
                SUFFIX_ORIGIN_DHCP,                 //< Provided by DHCP.
                SUFFIX_ORIGIN_LINK_LAYER_ADDRESS,   //< Obtained from layer 2.
                SUFFIX_ORIGIN_RANDOM                //< Obtained from random no.
            } SuffixOrigin;  

            /**
             * Create a UnicastAddressInformation with all-invalid properties.
             * All members are default initialised.
             */
            UnicastAddressInformation(void);

            /**
             * Clone 'rhs'.
             *
             * @param rhs The object to be cloned.
             */
            UnicastAddressInformation(const UnicastAddressInformation& rhs);

            /**
             * Dtor.
             */
            virtual ~UnicastAddressInformation(void);

            /**
             * Get the address itself.
             *
             * @return The address itself.
             */
            inline const IPAgnosticAddress& GetAddress(void) const {
                return this->address;
            }

            /**
             * Get the family of the address. 
             *
             * This is equivalent to calling GetAddress().GetAddressFamily().
             *
             * @return The address family.
             */
            inline IPAgnosticAddress::AddressFamily GetAddressFamily(
                    void) const {
                return this->address.GetAddressFamily();
            }

            /**
             * Answer the length of the address prefix.
             *
             * @outConfidence An optional pointer to a variable of type
             *                Confidence that receives the confidence
             *                for the value returned.
             *
             * @return The length of the address prefix in bits.
             *
             * @throws NoConfidenceException If the prefix length is invalid
             *                               and 'outConfidence' is NULL.
             */
            inline ULONG GetPrefixLength(
                    Confidence *outConfidence = NULL) const {
                this->prefixLength.GetConfidence(outConfidence, 
                    "Prefix Length");
                return this->prefixLength;
            }

            /**
             * Answer the source that the prefix was obtained from.
             *
             * @outConfidence An optional pointer to a variable of type
             *                Confidence that receives the confidence
             *                for the value returned.
             *
             * @return The prefix source.
             *
             * @throws NoConfidenceException If the prefix source is invalid 
             *                               and 'outConfidence' is NULL.
             */
            inline PrefixOrigin GetPrefixOrigin(
                    Confidence *outConfidence = NULL) const {
                this->prefixOrigin.GetConfidence(outConfidence, 
                    "Prefix Origin");
                return this->prefixOrigin;
            }

            /**
             * Answer the source that the suffix was obtained from.
             *
             * @outConfidence An optional pointer to a variable of type
             *                Confidence that receives the confidence
             *                for the value returned.
             *
             * @return The suffix source.
             *
             * @throws NoConfidenceException If the suffix source is invalid 
             *                               and 'outConfidence' is NULL.
             */
            inline SuffixOrigin GetSuffixOrigin(
                    Confidence *outConfidence = NULL) const {
                this->suffixOrigin.GetConfidence(outConfidence, 
                    "Suffix Origin");
                return this->suffixOrigin;
            }

            /**
             * Assignment.
             *
             * @param rhs The right hand side operand.
             *
             * @return *this.
             */
            UnicastAddressInformation& operator =(
                const UnicastAddressInformation& rhs);

            /**
             * Test for equality.
             *
             * @param rhs The right hand side operand.
             *
             * @return true if 'rhs' and this object are equal, 
             *         false otherwise.
             */
            bool operator ==(const UnicastAddressInformation& rhs) const;

            /**
             * Answer wether this object contains address information about the
             * given IPv4 address.
             *
             * @param rhs The right hand side operand.
             *
             * @return true if 'rhs' and this object are equal, 
             *         false otherwise.
             */
            bool operator ==(const IPAddress& rhs) const {
                return (this->address == rhs);
            }

            /**
             * Answer wether this object contains address information about the
             * given IPv6 address.
             *
             * @param rhs The right hand side operand.
             *
             * @return true if 'rhs' and this object are equal, 
             *         false otherwise.
             */
            bool operator ==(const IPAddress6& rhs) const {
                return (this->address == rhs);
            }

            /**
             * Answer wether this object contains address information about the
             * given IP address.
             *
             * @param rhs The right hand side operand.
             *
             * @return true if 'rhs' and this object are equal, 
             *         false otherwise.
             */
            bool operator ==(const IPAgnosticAddress& rhs) const {
                return (this->address == rhs);
            }

            /**
             * Test for inequality.
             *
             * @param rhs The right hand side operand.
             *
             * @return true if 'rhs' and this object are not equal, 
             *         false otherwise.
             */
            inline bool operator !=(
                    const UnicastAddressInformation& rhs) const {
                return !(*this == rhs);
            }

        private:

            /**
             * Initialise a new instance.
             *
             * @param endPoint               The endpoint to retrieve the 
             *                               address from.
             * @param prefixLength           The prefix length in bits.
             * @param prefixLengthConfidence Confidence of 'prefixLength'.
             * @param prefixOrigin           The origin of the address prefix.
             * @param prefixOriginConfidence Confidence of 'prefixOrigin'. 
             * @param suffixOrigin           The origin of the address suffix.
             * @param suffixOriginConfidence Confidence of 'suffixOrigin'. 
             */
            UnicastAddressInformation(const IPEndPoint endPoint, 
                const ULONG prefixLength, 
                const Confidence prefixLengthConfidence,
                const PrefixOrigin prefixOrigin,
                const Confidence prefixOriginConfidence,
                const SuffixOrigin suffixOrigin,
                const Confidence suffixOriginConfidence);

            /** The address itself. */
            IPAgnosticAddress address;

            /** Length of the prefix (equivalent of netmask). */
            AssessedMember<ULONG> prefixLength;

            /** Origin of the prefix. */
            AssessedMember<PrefixOrigin> prefixOrigin;

            /** Origin of the suffix. */
            AssessedMember<SuffixOrigin> suffixOrigin;

            /* Allow the NetworkInformation creating instances. */
            friend class NetworkInformation;

        }; /* end class UnicastAddressInformation */


        /** A list of IP addresses. */
        typedef Array<IPAgnosticAddress> AddressList;


        /** A list of unicast IP addresses. */
        typedef Array<UnicastAddressInformation> UnicastAddressList;


        /**
         * Nested class for adapter information.
         */
        class Adapter {

        public:

            // TODO: documentation
            typedef enum ScopeLevel_t  {
                SCOPE_INTERFACE = 1,
                SCOPE_LINK = 2,
                SCOPE_SUBNET = 3,
                SCOPE_ADMIN = 4,
                SCOPE_SITE = 5,
                SCOPE_ORGANISATION = 8,
                SCOPE_GLOBAL = 14
            } ScopeLevel;

            /**
             * This enumeration contains possible states of a network adapter. 
             * The members are derived from RFC 2863, see
             * http://www.ietf.org/rfc/rfc2863.txt
             */
            typedef enum OperStatus_t {
                OPERSTATUS_UNKNOWN = 0, //< Operational status is unknown.
                OPERSTATUS_UP,          //< Interface is up.
                OPERSTATUS_DOWN,        //< Interface is down.
                OPERSTATUS_TESTING,     //< Interface is in testing mode.
                OPERSTATUS_DORMANT,     //< Interface is waiting for events.
                OPERSTATUS_NOT_PRESENT, //< Refinement of STATUS_DOWN.
                OPERSTATUS_LOWER_LAYER_DOWN //< Refinement of STATUS_DOWN.
            } OperStatus;

            /**
             * The encapsulation method used by a tunnel if the adapter address 
             * is a tunnel. 
             * The members are as defined by the Internet Assigned Names 
             * Authority (IANA). For more information, see 
             * http://www.iana.org/assignments/ianaiftype-mib.
             */
            typedef enum TunnelType_t {
                TUNNELTYPE_NONE = 0,        //< Not a tunnel.
                TUNNELTYPE_OTHER = 1,       //< None of the following types.
                TUNNELTYPE_DIRECT = 2,      //< Packet encapsulated directly.
                TUNNELTYPE_6TO4 = 11,       //< Direct IPv6 in IPv4.
                TUNNEL_TYPE_ISATAP = 13,    //< IPv6 tunnel with ISATAP proto.
                TUNNEL_TYPE_TEREDO = 14     //< IPv6 Teredo encapsulation.
            } TunnelType; 

            /**
             * This enumeration contains the known types of network adapters. 
             * The members are as defined by the Internet Assigned Names 
             * Authority (IANA).
             */
            typedef enum Type_t {
                TYPE_OTHER = 0, //< Some other type of network interface.
                TYPE_ETHERNET,  //< An Ethernet network interface.
                TYPE_TOKENRING, //< A token ring network interface.
                TYPE_PPP,       //< A PPP network interface.
                TYPE_LOOPBACK,  //< A software loopback network interface.
                TYPE_ATM,       //< An ATM network interface.
                TYPE_IEEE80211, //< An IEEE 802.11 wireless network interface.
                TYPE_TUNNEL,    //< A tunnel encapsulation network interface.
                TYPE_IEEE1394   //< An IEEE 1394 serial bus network interface.
            } Type;

            /** 
             * Ctor. 
             */
            Adapter(void);

            /**
             * Clone 'rhs'.
             *
             * @param rhs The object to be cloned.
             */
            Adapter(const Adapter& rhs);

            /** Dtor. */
            ~Adapter(void);

            /**
             * Create a string from the physical (MAC) address. 
             * 
             * In case that the physical address is not valid, an empty string
             * is returned.
             *
             * @return A string representation of the physical address.
             */
            StringA FormatPhysicalAddressA(void) const;

            /**
             * Create a string from the physical (MAC) address. 
             * 
             * In case that the physical address is not valid, an empty string
             * is returned.
             *
             * @return A string representation of the physical address.
             */
            StringW FormatPhysicalAddressW(void) const;

            /** 
             * Answer the list of anycast addresses associated with the adapter.
             *
             * @outConfidence An optional pointer to a variable of type
             *                Confidence that receives the confidence
             *                for the value returned.
             *
             * @return The list of anycast addresses.
             *
             * @throws NoConfidenceException If the return value is invalid 
             *                               and 'outConfidence' is NULL.
             */
            inline const AddressList& GetAnycastAddresses(
                    Confidence *outConfidence = NULL) const {
                this->anycastAddresses.GetConfidence(outConfidence, 
                    "Anycast Addresses");
                return this->anycastAddresses;
            }

            /** 
             * Answer the IPv4 broadcast address of the adapter.
             *
             * @outConfidence An optional pointer to a variable of type
             *                Confidence that receives the confidence
             *                for the value returned.
             *
             * @return The broadcast address.
             *
             * @throws NoConfidenceException If the return value is invalid 
             *                               and 'outConfidence' is NULL.
             */
            inline const IPAddress& GetBroadcastAddress(
                    Confidence *outConfidence = NULL) const {
                this->broadcastAddress.GetConfidence(outConfidence, 
                    "Broadcast Address");
                return this->broadcastAddress;
            }

            /** 
             * Answer the human-readable description of the adapter.
             *
             * @outConfidence An optional pointer to a variable of type
             *                Confidence that receives the confidence
             *                for the value returned.
             *
             * @return The description string.
             *
             * @throws NoConfidenceException If the return value is invalid 
             *                               and 'outConfidence' is NULL.
             */
            inline const StringW& GetDescription(
                    Confidence *outConfidence = NULL) const {
                this->description.GetConfidence(outConfidence, 
                    "Description");
                return this->description;
            }

            /** 
             * Answer the ID of the adapter.
             *
             * On Windows, the name of the adapter can be changed by the user, 
             * but the ID remains the same.
             *
             * @outConfidence An optional pointer to a variable of type
             *                Confidence that receives the confidence
             *                for the value returned.
             *
             * @return The ID.
             *
             * @throws NoConfidenceException If the return value is invalid 
             *                               and 'outConfidence' is NULL.
             */
            inline const StringA& GetID(
                    Confidence *outConfidence = NULL) const {
                this->id.GetConfidence(outConfidence, "ID");
                return this->id;
            }

            /** 
             * Answer the maximum transfer unit of the adapter.
             *
             * @outConfidence An optional pointer to a variable of type
             *                Confidence that receives the confidence
             *                for the value returned.
             *
             * @return The MTU.
             *
             * @throws NoConfidenceException If the return value is invalid 
             *                               and 'outConfidence' is NULL.
             */
            inline UINT GetMTU(Confidence *outConfidence = NULL) const {
                this->mtu.GetConfidence(outConfidence, "MTU");
                return this->mtu;
            }

            /** 
             * Answer the list of multicast addresses associated with the 
             * adapter.
             *
             * @outConfidence An optional pointer to a variable of type
             *                Confidence that receives the confidence
             *                for the value returned.
             *
             * @return The list of multicast addresses.
             *
             * @throws NoConfidenceException If the return value is invalid 
             *                               and 'outConfidence' is NULL.
             */
            inline const AddressList& GetMulticastAddresses(
                    Confidence *outConfidence = NULL) const {
                this->multicastAddresses.GetConfidence(outConfidence, 
                    "Multicast Addresses");
                return this->multicastAddresses;
            }

            /** 
             * Answer the friendly name of the adapter.
             *
             * @outConfidence An optional pointer to a variable of type
             *                Confidence that receives the confidence
             *                for the value returned.
             *
             * @return The name of the adapter.
             *
             * @throws NoConfidenceException If the return value is invalid 
             *                               and 'outConfidence' is NULL.
             */
            inline const StringW& GetName(
                    Confidence *outConfidence = NULL) const {
                this->name.GetConfidence(outConfidence, "Name");
                return this->name;
            }

            /** 
             * Answer the physical (MAC) address of the adapter.
             *
             * @outConfidence An optional pointer to a variable of type
             *                Confidence that receives the confidence
             *                for the value returned.
             *
             * @return The raw MAC address.
             *
             * @throws NoConfidenceException If the MAC address is invalid 
             *                               and 'outConfidence' is NULL.
             */
            const Array<BYTE>& GetPhysicalAddress(
                Confidence *outConfidence = NULL) const;

            /** 
             * Answer the current status of the adapter. This is one of the 
             * values defined in RFC 2863.
             *
             * On Linux, adapters are always up, because disabled devices
             * cannot be enumerated as of now.
             *
             * @outConfidence An optional pointer to a variable of type
             *                Confidence that receives the confidence
             *                for the value returned.
             *
             * @return The current status of the adapter.
             *
             * @throws NoConfidenceException If the return value is invalid 
             *                               and 'outConfidence' is NULL.
             */
            inline OperStatus GetStatus(
                    Confidence *outConfidence = NULL) const {
                this->status.GetConfidence(outConfidence, "Status");
                return this->status;
            }

            /** 
             * Answer the type of the adapter.
             *
             * @outConfidence An optional pointer to a variable of type
             *                Confidence that receives the confidence
             *                for the value returned.
             *
             * @return The type of the adapter.
             *
             * @throws NoConfidenceException If the return value is invalid 
             *                               and 'outConfidence' is NULL.
             */
            inline Type GetType(Confidence *outConfidence = NULL) const {
                this->name.GetConfidence(outConfidence, "Type");
                return this->type;
            }

            /**
             * Convenience method for getting one unicast address of the 
             * adapter. The method tries to find the first address having the
             * specified family. If no such address is found, the first address
             * is returned.
             *
             * @param preferredFamily The preferred address family returned.
             *
             * @return A unicast address of the adapter.
             *
             * @throws NoConfidenceException If the adapter has no valid
             *                               unicast addresses.
             * @throws OutOfRangeException If the adapter has no unicast 
             *                             addresses assigned.
             */
            const IPAgnosticAddress GetUnicastAddress(
                const IPAgnosticAddress::AddressFamily preferredFamily) 
                const;

            /** 
             * Answer the list of unicast addresses associated with the 
             * adapter.
             *
             * Note: If you are not sure what you are looking for, this is 
             * probably what you want.
             *
             * @outConfidence An optional pointer to a variable of type
             *                Confidence that receives the confidence
             *                for the value returned.
             *
             * @return The list of unicast addresses.
             *
             * @throws NoConfidenceException If the return value is invalid 
             *                               and 'outConfidence' is NULL.
             */
            inline const UnicastAddressList& GetUnicastAddresses(
                    Confidence *outConfidence = NULL) const {
                this->unicastAddresses.GetConfidence(outConfidence, 
                    "Unicast Addresses");
                return this->unicastAddresses;
            }

            /**
             * Assignment.
             *
             * @param rhs The right hand side operand.
             *
             * @return *this.
             */
            Adapter& operator =(const Adapter& rhs);

            /**
             * Test for equality.
             *
             * @param rhs The right hand side operand.
             *
             * @return true if this object and 'rhs' are equal,
             *         false otherwise.
             */
            bool operator ==(const Adapter& rhs) const;

            /**
             * Test for inequality.
             *
             * @param rhs The right hand side operand.
             *
             * @return true if this object and 'rhs' are not equal,
             *         false otherwise.
             */
            inline bool operator !=(const Adapter& rhs) const {
                return !(*this == rhs);
            }

        private:


            /** The anycast addresses of the adapter. */
            AssessedMember<AddressList> anycastAddresses;

            /** The IPv4 broadcast address of the adapter. */
            AssessedMember<IPAddress> broadcastAddress;

            /** A description of the adapter (human-readable device name). */
            AssessedMember<StringW> description;

            /** The permanent name (ID) of the adapter. */
            AssessedMember<StringA> id;

            /** The maximum transmission unit (MTU) size, in bytes. */
            AssessedMember<UINT> mtu;

            /** The multicast addressess of the adapter. */
            AssessedMember<AddressList> multicastAddresses;

            /** The human-readable adapter name. */
            AssessedMember<StringW> name;

            /** The physical MAC address of the adapter. */
            Array<BYTE> physicalAddress;

            /** The status of the adapter. */
            AssessedMember<OperStatus> status;

            /** The type of adapter. */
            AssessedMember<Type> type;

            /** The unicast addresses of the adapter. */
            AssessedMember<UnicastAddressList> unicastAddresses;

            /** Friend class for initializing the values. */
            friend class NetworkInformation;

        }; /* end class Adapter */

        /**
         * Callback function for EnumerateAdapters.
         * 
         * @param adapter     The currently enumerated adapter.
         * @param userContext A user-provided context pointer.
         *
         * @return true in order continue enumeration, false otherwise.
         */
        typedef bool (* EnumerateAdaptersCallback)(const Adapter& adapter, 
            void *userContext);

        /**
         * Callback function for selecting an adapter.
         *
         * @param adapter The adapter to assess now.
         * @param userContext A user-provided context pointer.
         *
         * @return true to select the adapter, false otherwise.
         */
        typedef bool (* SelectAdapterCallback)(const Adapter& adapter, 
            void *userContext);

        /** The list of adapters. */
        typedef Array<Adapter> AdapterList;

        /** 
         * Answer the number of known network adapters. 
         *
         * This method is thread-safe. Please note, that the results returned
         * might become invalid after the method returned in case that another
         * thread updates the adapter cache.
         *
         * @return The number of known adapters.
         * 
         * @throws SystemException If the adapters could not be retrieved due to
         *                         an error in a system call.
         * @throws SocketException If the adapters could not be retrieved due to
         *                         an error in a socket operation.
         * @throws std::bad_alloc  If the memory required to retrieve the 
         *                         adapters could not be allocated.
         */
        static SIZE_T CountAdapters(void);

        /**
         * Discard all cached adapter information.
         *
         * @param reread If set true, the cache is immediately re-read after 
         *               clearing.
         *
         * @throws SystemException If the adapters could not be retrieved due to
         *                         an error in a system call.
         * @throws SocketException If the adapters could not be retrieved due to
         *                         an error in a socket operation.
         * @throws std::bad_alloc  If the memory required to retrieve the 
         *                         adapters could not be allocated.
         */
        static void DiscardCache(const bool reread = false);

        /**
         * Enumerate all adapters by calling 'cb' for each known adapter.
         *
         * This method is thread-safe.
         *
         * @param cb
         * @param userContext
         *
         * @throws IllegalParamException If 'cb' is a NULL pointer.
         * @throws SystemException       If the adapters could not be retrieved 
         *                               due to an error in a system call.
         * @throws SocketException       If the adapters could not be retrieved 
         *                               due to an error in a socket operation.
         * @throws std::bad_alloc        If the memory required to retrieve the 
         *                               adapters could not be allocated.
         */
        static void EnumerateAdapters(EnumerateAdaptersCallback cb,
            void *userContext = NULL);

        /**
         * Answer the 'idx'th adapter of the system.
         *
         * This method is thread-safe.
         *
         * Please note that the method returns a deep copy of the adapter and 
         * therefore is less efficient than other methods. This is required for
         * ensuring thread-safety. You can use GetAdapterUnsafe() for faster,
         * but unsafe, access to an adapter.
         *
         * @param idx The index of the adapter to be retrieved. This must be
         *            within [0, NetworkInformation::CountAdapters()[.
         *
         * @return The 'idx'th adapter.
         *
         * @throws OutOfRangeException If 'idx' is not a valid adapter index.
         * @throws SystemException     If an error occurred while retrieving 
         *                             information from the OS.
         * @throws SocketException     If the socket subsystem could not be 
         *                             used.
         * @throws std::bad_alloc      If there was insufficient memory for 
         *                             retrieving the data.
         */
        static Adapter GetAdapter(const SIZE_T idx);

        /**
         * Answer the adapter with the given ID into 'outAdapter.
         *
         * This method is thread-safe.
         *
         * @param outAdapter Receives the adapter, if true is returned.
         * @param id         The ID (unique, unchangable name) of the adapter to
         *                   be retrieved.
         *
         * @return true if an adapter with the given ID was found,
         *         false otherwise.
         *
         * @throws SystemException     If an error occurred while retrieving 
         *                             information from the OS.
         * @throws SocketException     If the socket subsystem could not be 
         *                             used.
         * @throws std::bad_alloc      If there was insufficient memory for 
         *                             retrieving the data.
         */
        static bool GetAdapterForID(Adapter& outAdapter, const char *id);

        /**
         * Answer all adapters that fulfill the predicate implemented by 'cb'.
         *
         * This method is thread-safe.
         *
         * @param outAdapter  Receives the adapter, if true is returned.
         * @param cb          A callback function determining whether the adapter
         *                    shall be returned or not. If the method returns 
         *                    true for an adapter, it is returned, if it returns
         *                    false, it is ignored.
         * @param userContext
         * 
         * @return The number of adapters found.
         *
         * @throws SystemException     If an error occurred while retrieving 
         *                             information from the OS.
         * @throws SocketException     If the socket subsystem could not be 
         *                             used.
         * @throws std::bad_alloc      If there was insufficient memory for 
         *                             retrieving the data.
         */
        static SIZE_T GetAdaptersForPredicate(AdapterList& outAdapters,
            SelectAdapterCallback cb, void *userContext = NULL);

        /**
         * Answer all adapters of the given type
         *
         * This method is thread-safe.
         *
         * @param outAdapter Receives the adapter, if true is returned.
         * @param type       The type of adapters to be retrieved.
         * 
         * @return The number of adapters found.
         *
         * @throws SystemException     If an error occurred while retrieving 
         *                             information from the OS.
         * @throws SocketException     If the socket subsystem could not be 
         *                             used.
         * @throws std::bad_alloc      If there was insufficient memory for 
         *                             retrieving the data.
         */
        static SIZE_T GetAdaptersForType(AdapterList& outAdapters,
            const Adapter::Type type);

        /**
         * Answer all adapters that are bound to the given IP address.
         *
         * This method is thread-safe.
         *
         * @param outAdapters Receives the adapters.
         * @param address     The IP address that the adapter must have.
         *
         * @return The number of adapters found.
         *
         * @throws SystemException     If an error occurred while retrieving 
         *                             information from the OS.
         * @throws SocketException     If the socket subsystem could not be 
         *                             used.
         * @throws std::bad_alloc      If there was insufficient memory for 
         *                             retrieving the data.
         */
        static SIZE_T GetAdaptersForUnicastAddress(AdapterList& outAdapters, 
            const IPAddress& address);

        /**
         * Answer all adapters that are bound to the given IP address.
         *
         * This method is thread-safe.
         *
         * @param outAdapters Receives the adapters.
         * @param address     The IP address that the adapter must have.
         *
         * @return The number of adapters found.
         *
         * @throws SystemException     If an error occurred while retrieving 
         *                             information from the OS.
         * @throws SocketException     If the socket subsystem could not be 
         *                             used.
         * @throws std::bad_alloc      If there was insufficient memory for 
         *                             retrieving the data.
         */
        static SIZE_T GetAdaptersForUnicastAddress(AdapterList& outAdapters, 
            const IPAddress6& address);

        /**
         * Answer all adapters that are bound to the given IP address.
         *
         * This method is thread-safe.
         *
         * @param outAdapters Receives the adapters.
         * @param address     The IP address that the adapter must have.
         *
         * @return The number of adapters found.
         *
         * @throws SystemException     If an error occurred while retrieving 
         *                             information from the OS.
         * @throws SocketException     If the socket subsystem could not be 
         *                             used.
         * @throws std::bad_alloc      If there was insufficient memory for 
         *                             retrieving the data.
         */
        static SIZE_T GetAdaptersForUnicastAddress(AdapterList& outAdapters, 
            const IPAgnosticAddress& address);

        /**
         * Answer all adapters that have the same prefix of length
         * 'prefixLength' as the given IP address 'address'.
         *
         * This method is thread-safe.
         *
         * @param outAdapters Receives the adapters.
         * @param address     The IP address that the adapter must have.
         *
         * @return The number of adapters found.
         *
         * @throws SystemException     If an error occurred while retrieving 
         *                             information from the OS.
         * @throws SocketException     If the socket subsystem could not be 
         *                             used.
         * @throws std::bad_alloc      If there was insufficient memory for 
         *                             retrieving the data.
         */
        static SIZE_T GetAdaptersForUnicastPrefix(AdapterList& outAdapters,
            const IPAgnosticAddress& address, const ULONG prefixLength);

        /**
         *
         * This method is thread-safe with regard to the update of the adapter
         * list. Once the reference to the list is returned, the list is not
         * protected any more.
         */
        //static const AdapterList& GetAdaptersUnsafe(
        //    const bool forceUpdate = false);

        /** 
         * Answer the 'idx'th adapter of the system.
         *
         * This method is thread-safe with regard to the update of the adapter
         * list. Access to the reference returned is not safe any more and may
         * result in a dangling reference if the adapter cache is updated.
         *
         * @param idx The index of the adapter to be retrieved. This must be
         *            within [0, NetworkInformation::CountAdapters()[.
         *
         * @return The 'idx'th adapter.
         *
         * @throws OutOfRangeException If 'idx' is not a valid adapter index.
         * @throws SystemException     If an error occurred while retrieving 
         *                             information from the OS.
         * @throws SocketException     If the socket subsystem could not be 
         *                             used.
         * @throws std::bad_alloc      If there was insufficient memory for 
         *                             retrieving the data.
         */
        static const Adapter& GetAdapterUnsafe(const SIZE_T idx);

        /**
         * Performs a wild guess on which adapter could be designated by the 
         * given string representation 'str'.
         *
         * The string representation that can be parsed assumes the following
         * basic format:
         *
         * <IP/host name/device name>/<netmask/prefix length>:<port>
         *
         * The port number is ignored for guessing the adapter and is only 
         * recognised for convenience.
         *
         * The IP address, host name and device name are mutually exclusive. The
         * best interpretation will be used. IP addresses can be either IPv4 or
         * IPv6. IPv6 addresses can be enclosed in brackets ("[", "]").
         *
         * The netmask (including the preceding "/") is optional.
         *
         * An IP address can include an optional zone ID ("%<zone>"), which is
         * not used for guessing.
         *
         * If you specify an IP address, host or device name that exactly 
         * matches an adapter and you also specify a prefix length which is
         * not applicable for any IP address of the designated adapter, the
         * wildness of the guess is increased.
         *
         * The wildness of the guess is increased for adapters that are 
         * currently down.
         *
         * @param outAdapter     Receives the guessed adapter.
         * @param str            The string that should identify an adapter.
         * @param invertWildness Return (1.0 - wildness) instead of the 
         *                       wildness. Note that this is not chefm‰ﬂig!
         * 
         * @return The wildness, which is a value within [0, 1] indicating "how
         *         wild" the guess was. If the value is 0, the result can be
         *         considered correct and unambiguous. If it is near 1, the 
         *         result is mostly random. If 'inverWildness' is set true, the
         *         return value is inverted, i.e. 1.0 in best case and 0.0 in
         *         worst case.
         *
         * @throws NoSuchElementException If there is no known adapter 
         *                                available.
         */
        static float GuessAdapter(Adapter& outAdapter, const char *str,
                const bool invertWildness = false);

        /**
         * Performs a wild guess on which adapter could be designated by the 
         * given string representation 'str'.
         *
         * The string representation that can be parsed assumes the following
         * basic format:
         *
         * <IP/host name/device name>/<netmask/prefix length>:<port>
         *
         * The port number is ignored for guessing the adapter and is only 
         * recognised for convenience.
         *
         * The IP address, host name and device name are mutually exclusive. The
         * best interpretation will be used. IP addresses can be either IPv4 or
         * IPv6. IPv6 addresses can be enclosed in brackets ("[", "]").
         *
         * The netmask (including the preceding "/") is optional.
         *
         * An IP address can include an optional zone ID ("%<zone>"), which is
         * not used for guessing.
         *
         * If you specify an IP address, host or device name that exactly 
         * matches an adapter and you also specify a prefix length which is
         * not applicable for any IP address of the designated adapter, the
         * wildness of the guess is increased.
         *
         * The wildness of the guess is increased for adapters that are 
         * currently down.
         *
         * @param outAdapter     Receives the guessed adapter.
         * @param str            The string that should identify an adapter.
         * @param invertWildness Return (1.0 - wildness) instead of the 
         *                       wildness. Note that this is not chefm‰ﬂig!
         * 
         * @return The wildness, which is a value within [0, 1] indicating "how
         *         wild" the guess was. If the value is 0, the result can be
         *         considered correct and unambiguous. If it is near 1, the 
         *         result is mostly random. If 'inverWildness' is set true, the
         *         return value is inverted, i.e. 1.0 in best case and 0.0 in
         *         worst case.
         *
         * @throws NoSuchElementException If there is no known adapter 
         *                                available.
         */
        static float GuessAdapter(Adapter& outAdapter, const wchar_t *str,
            const bool invertWildness = false);

        // TODO: documentation
        static float GuessLocalEndPoint(IPEndPoint& outEndPoint, 
            const char *str, 
            const IPAgnosticAddress::AddressFamily preferredFamily, 
            const bool invertWildness = false);
    
        // TODO: documentation
        static float GuessLocalEndPoint(IPEndPoint& outEndPoint, 
            const wchar_t *str, 
            const IPAgnosticAddress::AddressFamily preferredFamily, 
            const bool invertWildness = false);

        // TODO: documentation
        static float GuessLocalEndPoint(IPEndPoint& outEndPoint, 
            const char *str, const bool invertWildness = false);
    
        // TODO: documentation
        static float GuessLocalEndPoint(IPEndPoint& outEndPoint, 
            const wchar_t *str, const bool invertWildness = false);

        // TODO: documentation
        static float GuessRemoteEndPoint(IPEndPoint& outEndPoint, 
            const char *str, 
            const IPAgnosticAddress::AddressFamily preferredFamily,
            const bool invertWildness = false);

        // TODO: documentation
        static float GuessRemoteEndPoint(IPEndPoint& outEndPoint, 
            const char *str, const bool invertWildness = false);

        // TODO: documentation
        static float GuessRemoteEndPoint(IPEndPoint& outEndPoint, 
            const wchar_t *str, 
            const IPAgnosticAddress::AddressFamily preferredFamily, 
            const bool invertWildness = false);

        // TODO: documentation
        static float GuessRemoteEndPoint(IPEndPoint& outEndPoint, 
            const wchar_t *str, const bool invertWildness = false);

        /**
         * Convert a network mask to a prefix length.
         *
         * @param netmask The netmask to convert.
         *
         * @return The equivalent prefix length in bits.
         *
         * @throws IllegalParamException If 'netmask' is not a valid netmask.
         */
        inline static ULONG NetmaskToPrefix(const IPAddress& netmask) {
            const BYTE *mask = reinterpret_cast<const BYTE *>(
                static_cast<const struct in_addr *>(netmask));
            return NetworkInformation::netmaskToPrefix(mask, 
                sizeof(struct in_addr));
        }

        /**
         * Convert a network mask to a prefix length.
         *
         * @param netmask The netmask to convert.
         *
         * @return The equivalent prefix length in bits.
         *
         * @throws IllegalParamException If 'netmask' is not a valid netmask.
         */
        inline static ULONG NetmaskToPrefix(const IPAddress6& netmask) {
            const BYTE *mask = reinterpret_cast<const BYTE *>(
                static_cast<const struct in6_addr *>(netmask));
            return NetworkInformation::netmaskToPrefix(mask, 
                sizeof(struct in6_addr));
        }

        /**
         * Convert a network mask to a prefix length.
         *
         * @param netmask The netmask to convert.
         *
         * @return The equivalent prefix length in bits.
         *
         * @throws IllegalParamException If 'netmask' is not a valid netmask.
         */
        static ULONG NetmaskToPrefix(const IPAgnosticAddress& netmask);

        /**
         * Convert a prefix length to an IPv4 netmask.
         *
         * @param prefix The prefix length in bits.
         *
         * @return The equivalent netmask.
         *
         * @throws OutOfRangeException If the prefix length is not within the 
         *                             valid range of an IPv4 address.
         */
        inline static IPAddress PrefixToNetmask4(const ULONG prefix) {
            IPAddress retval;
            BYTE *mask = reinterpret_cast<BYTE *>(
                static_cast<struct in_addr *>(retval));
            NetworkInformation::prefixToNetmask(mask, sizeof(struct in_addr), 
                prefix);
            return retval;
        }

        /**
         * Answer a human-readable string for the given scope level 
         * enumeration value.
         *
         * @param sl The enumeration value to be stringised.
         *
         * @return A string representing the enumeration value. The memory
         *         is owned by the callee.
         *
         * @throws IllegalParamException If the enumeration value cannot be
         *                               stringised.
         */
        static const char *Stringise(const Adapter::ScopeLevel sl);

        /**
         * Answer a human-readable string for the given adapter status
         * enumeration value.
         *
         * @param as The enumeration value to be stringised.
         *
         * @return A string representing the enumeration value. The memory
         *         is owned by the callee.
         *
         * @throws IllegalParamException If the enumeration value cannot be
         *                               stringised.
         */
        static const char *Stringise(const Adapter::OperStatus as);

        /**
         * Answer a human-readable string for the given adapter type
         * enumeration value.
         *
         * @param at The enumeration value to be stringised.
         *
         * @return A string representing the enumeration value. The memory
         *         is owned by the callee.
         *
         * @throws IllegalParamException If the enumeration value cannot be
         *                               stringised.
         */
        static const char *Stringise(const Adapter::Type at);

        /**
         * Answer a human-readable string for the given tunnel type
         * enumeration value.
         *
         * @param at The enumeration value to be stringised.
         *
         * @return A string representing the enumeration value. The memory
         *         is owned by the callee.
         *
         * @throws IllegalParamException If the enumeration value cannot be
         *                               stringised.
         */
        static const char *Stringise(const Adapter::TunnelType tt);

        /**
         * Answer a human-readable string for the given address prefix origin
         * enumeration value.
         *
         * @param po The enumeration value to be stringised.
         *
         * @return A string representing the enumeration value. The memory
         *         is owned by the callee.
         *
         * @throws IllegalParamException If the enumeration value cannot be
         *                               stringised.
         */
        static const char *Stringise(
            const UnicastAddressInformation::PrefixOrigin po);

        /**
         * Answer a human-readable string for the given address suffix origin
         * enumeration value.
         *
         * @param so The enumeration value to be stringised.
         *
         * @return A string representing the enumeration value. The memory
         *         is owned by the callee.
         *
         * @throws IllegalParamException If the enumeration value cannot be
         *                               stringised.
         */
        static const char *Stringise(
            const UnicastAddressInformation::SuffixOrigin so);

        /**
         * Discard all cached adapter information and re-read it.
         *
         * @throws SystemException
         * @throws SocketException
         * @throws std::bad_alloc
         */
        inline static void Update(void) {
            NetworkInformation::DiscardCache(true);
        }

    private:

        /**
         * This is the context structure for enumerating adapters using the 
         * callback function processAdapterForLocalEndpointGuess.
         */
        typedef struct GuessLocalEndPointCtx_t {
            IPAgnosticAddress *Address; //< Pointer to input address.
            ULONG *PrefixLen;           //< Pointer to input prefix length.
            Array<float> *Wildness;     //< Output array of address wildness.
            bool IsIPv4Preferred;       //< Prefer IPv4 in case of doubt.
        } GuessLocalEndPointCtx;

        // TODO: documentation
        static float assessAddressAsEndPoint(
            const UnicastAddressInformation& addrInfo, 
            const GuessLocalEndPointCtx& ctx);

        /**
         * Enforces that the wildness in 'inOutWildness' is within [0, 1] and 
         * find the wildness that is the lowest wildness in this array. The
         * method returns the index of this wildness to 'outIdxFirstBest' and
         * the actual numeric value as return value.
         *
         * As a side-effect of the method, all values in 'inOutWildness' will
         * be forced to [0, 1].
         *
         * TODO: Should we normalise the wildness range rather than clamping
         * it? Should we preserver 0 and 1 in this case?
         *
         * @param inOutWildness   A list of wildness values.
         * @param outIdxFirstBest The index of the minimum value in 
         *                        'inOutWildness'. If the minimum occurs 
         *                        multiple times, the first occurrence is 
         *                        returned.
         *
         * @return The minimum wildness found. If the minimum occurs mutliple
         *         times, it will be weighted except for 0 and 1. This weighting
         *         change is not reflected in 'inOutWildness'.
         */
        static float consolidateWildness(Array<float>& inOutWildness, 
            SIZE_T& outIdxFirstBest);

        /**
         * Answer whether the UnicastAddressList 'list' contains the IPAddress,
         * IPAddress6 or IPAgnosticAddress 'addr'. 
         *
         * @param list   The list to be searched for 'addr'.
         * @param addr   The address to be searched.
         * @param startIdx  First index to check in 'list'.
         *
         * @return The index of the first occurrence of 'addr' in 'list',
         *         or -1 if not found.
         */
        template<class A> 
        static int findAddress(const UnicastAddressList& list, 
            const A& addr, const int startIdx = 0);

        /**
         * Answer whether the UnicastAddressList 'list' contains an address that
         * has the same prefix as specified by the IPAddress, IPAddress6 or
         * IPAgnosticAddress 'addr' and the prefix length 'prefixLen'.
         *
         *
         * @param list      The list to be searched for the prefix.
         * @param addr      The address defining the prefix.
         * @param prefixLen The prefix length.
         * @param startIdx  First index to check in 'list'.
         *
         * @return The index of the first address with the given prefix in 
         *         'list', or -1 if not found.
         */
        template<class A> 
        static int findSamePrefix(const UnicastAddressList& list, 
            const A& addr, const ULONG prefixLen, const int startIdx = 0);

        /**
         * Guess a broadcast address for the given adapter address 'address' 
         * and the network mask 'netmask'.
         *
         * @param address The address to guess the broadcast address for.
         * @param netmask The network mast of the adapter.
         *
         * @return A guess for the broadcast address.
         *
         * @throws IllegalParamException If no guess could be made, e.g. 
         *                               because the netmask is empty.
         */
        static IPAddress guessBroadcastAddress(const IPAddress& address, 
            const IPAddress& netmask);

        /**
         * Implementation of the GuessLocalEndPoint functionality. This 
         * method handles all the work.
         *
         * @param outEndPoint
         * @param str
         * @param prefFam        If the input string does not contain any
         *                       information to deduce the address family
         *                       from, use this address family (if not NULL).
         * @param invertWildness
         *
         * @return
         */
        static float guessLocalEndPoint(IPEndPoint& outEndPoint, 
            const wchar_t *str, const IPAgnosticAddress::AddressFamily *prefFam,
            const bool invertWildness);

        /**
         * Implementation of the GuessRemoteEndPoint functionality. This 
         * method handles all the work.
         *
         * @param outEndPoint
         * @param str
         * @param prefFam        If the input string does not contain any
         *                       information to deduce the address family
         *                       from, use this address family (if not NULL).
         * @param invertWildness
         *
         * @return
         */
        static float guessRemoteEndPoint(IPEndPoint& outEndPoint, 
            const wchar_t *str, const IPAgnosticAddress::AddressFamily *prefFam,
            const bool invertWildness);

        /**
         * Initializes the list of network adapter objects.
         *
         * This method performs an initialisation of the 
         * NetworkInformation::adapters member if either the adapter list is 
         * empty.
         *
         * This method is not thread-safe!
         *
         * @throws SystemException If an error occurred while retrieving 
         *                         information from the OS.
         * @throws SocketException If the socket subsystem could not be used.
         * @throws std::bad_alloc  If there was insufficient memory for 
         *                         retrieving the data.
         */
        static void initAdapters(void);

        /**
         * Maps a system defined interface type constant to the VISlib
         * Adapter::Type enumeration.
         *
         * @param prefixOrigin The constant to be mapped.
         *
         * @return The VISlib equivalent of the input.
         *
         * @throws IllegalParamException If no valid mapping could be found.
         */
#ifdef _WIN32
        static Adapter::Type mapAdapterType(const IFTYPE type);
#else /* _WIN32 */
        static Adapter::Type mapAdapterType(const int type);
#endif /* _WIN32 */

#ifdef _WIN32
        /**
         * Maps a system defined operation state constant to the VISlib
         * Adapter::OperStatus enumeration.
         *
         * @param prefixOrigin The constant to be mapped.
         *
         * @return The VISlib equivalent of the input.
         *
         * @throws IllegalParamException If no valid mapping could be found.
         */
        static Adapter::OperStatus mapOperationStatus(
            const IF_OPER_STATUS status);
#endif /* _WIN32 */

#ifdef _WIN32
        /**
         * Map the system defined prefix origin constant to the corresponding
         * VISlib enumeration type.
         *
         * @param prefixOrigin The constant to be mapped.
         *
         * @return The VISlib equivalent of the input.
         *
         * @throws IllegalParamException If no valid mapping could be found.
         */
        static UnicastAddressInformation::PrefixOrigin mapPrefixOrigin(
            const IP_PREFIX_ORIGIN prefixOrigin);

        /**
         * Map the system defined suffix origin constant to the corresponding
         * VISlib enumeration type.
         *
         * @param prefixOrigin The constant to be mapped.
         *
         * @return The VISlib equivalent of the input.
         *
         * @throws IllegalParamException If no valid mapping could be found.
         */
        static UnicastAddressInformation::SuffixOrigin mapSuffixOrigin(
            const IP_SUFFIX_ORIGIN suffixOrigin);
#endif /* _WIN32 */

        /**
         * Convert a network mask to a prefix length.
         *
         * @param netmask The netmask to convert.
         * @param len     The length of the netmask in bytes.
         *
         * @return The equivalent prefix length in bits.
         *
         * @throws IllegalParamException If 'netmask' is not a valid netmask.
         */
        static ULONG netmaskToPrefix(const BYTE *netmask, const SIZE_T len);

        /**
         * Convert a prefix length to a netmask.
         *
         * @param outNetmask Raw storage of the network mask.
         * @param len        Size of the 'outNetmask' buffer.
         * @param prefix     The prefix length in bits.
         *
         * @return The equivalent netmask.
         *
         * @throws OutOfRangeException If the prefix length is not within the 
         *                             valid range.
         */
        static void prefixToNetmask(BYTE *outNetmask, const SIZE_T len,
            const ULONG prefix);

        /**
         * Process an enumerated adapter for guessing a local endpoint.
         *
         * The method adds the wildness for the adatper to the 'Wildness' array
         * in the context structure.
         *
         * @param adapter The currently processed adapter.
         * @param context A pointer to a GuessLocalEndPointCtx that must live
         *                until the method returns.
         *
         * @return true
         */
        static bool processAdapterForLocalEndpointGuess(const Adapter& adapter, 
            void *context);

        /**
         * Answer whether the adapter 'adapter' has given type.
         *
         * @param adapter The adapter to be checked.
         * @param addr    Pointer to an Adapter::Type.
         *
         * @return true if the type has been found, false otherwise.
         */
        static bool selectAdapterByType(const Adapter& adapter, void *type);

        /**
         * Answer whether the adapter 'adapter' has the IPAgnosticAddress 'addr'
         * in its unicast address list.
         *
         * @param adapter The adapter to be checked.
         * @param addr    Pointer to an IPAgnosticAddress.
         *
         * @return true if the address has been found, false otherwise.
         */
        static bool selectAdapterByUnicastIP(const Adapter& adapter, 
            void *addr);

        /**
         * Answer whether the adapter 'adapter' has the IPAddress 'addr'
         * in its unicast address list.
         *
         * @param adapter The adapter to be checked.
         * @param addr    Pointer to an IPAddress.
         *
         * @return true if the address has been found, false otherwise.
         */
        static bool selectAdapterByUnicastIPv4(const Adapter& adapter, 
            void *addr);

        /**
         * Answer whether the adapter 'adapter' has the IPAddress6 'addr'
         * in its unicast address list.
         *
         * @param adapter The adapter to be checked.
         * @param addr    Pointer to an IPAddress6.
         *
         * @return true if the address has been found, false otherwise.
         */
        static bool selectAdapterByUnicastIPv6(const Adapter& adapter, 
            void *addr);

        /**
         * Answer whether the adapter 'adapter' is in the same subnet
         * as specified by the UnicastAddressInformation 'addrInfo' and its
         * address and prefix length.
         *
         * @param adapter  The adapter to be checked.
         * @param addrInfo Pointer to a UnicastAddressInformation.
         *
         * @return true if the address has been found, false otherwise.
         */
        static bool selectAdapterByUnicastPrefix(const Adapter& adapter,
            void *addrInfo);

        /**
         * Implements the wild guess on an adapter based on the parsed input
         * string.
         *
         * The guessing rules are as described for GuessAdapter().
         *
         * This operation is not thread-safe! Lock outside!
         *
         * @param outAdapter Receives the adapter.
         * @param address    The address of the adapter to search.
         * @param device     The device name of the adapter to search.
         * @param prefixLen  The prefix length of the subnet.
         * @param validMask  Specifies which of the input variables are valid.
         *
         * @return The wildness of the guess within [0, 1].
         */
        static float wildGuessAdapter(Adapter& outAdapter, 
            const IPAgnosticAddress& address, const StringW& device, 
            const ULONG prefixLen, const UINT32 validMask);

        /**
         * Split the input on behalf of the wild guess method into the parts
         * that are used to identify an adapter or end point. 
         * 
         * The splitting rules are as described for GuessAdapter().
         *
         * @param outAddress   Receives the address if specified in 'str'.
         * @param outDevice    Receives the device name if specified in 'str'. 
         * @param outPrefixLen Receives the prefix length if specified in 'str'.
         * @param outPort      Receives the port if any specified in 'str'.
         * @param str          The string to be checked.
         * @param prefFam      Initialise the preferred address family with
         *                     this value. If NULL, use IPv6. The preferred
         *                     family might change based on the user input. If
         *                     the user input does not give any hints about the
         *                     preferred address family, this value is used.
         *
         * @return A bitmask specifying which of the out parameters are valid.
         */
        static UINT32 wildGuessSplitInput(IPAgnosticAddress& outAddress,
            StringW& outDevice, ULONG& outPrefixLen, USHORT& outPort,
            const wchar_t *str, 
            const IPAgnosticAddress::AddressFamily *prefFam = NULL);

        /**
         * Wildness penalty for an adapter that matches, but is currently down.
         */
        static const float PENALTY_ADAPTER_DOWN;

        /**
         * Wildness penalty for an empty address, which might be parsed 
         * successfully.
         */
        static const float PENALTY_EMPTY_ADDRESS;

        /** Wildness penalty for missing port. */
        static const float PENALTY_NO_PORT;

        /**
         * Wildness penalty for a wrong address family.
         */
        static const float PENALTY_WRONG_ADDRESSFAMILY;

        /**
         * Wildness penalty for a wrong prefix while the rest of the adapter
         * specification matches.
         */
        static const float PENALTY_WRONG_PREFIX;

        /**
         * Flag indicating that the address returned by wildGuessSplitInput()
         * was parsed from an empty string.
         */
        static const UINT32 WILD_GUESS_FROM_EMPTY_ADDRESS;

        /**
         * Flag that is returned by wildGuessSplitInput() indicating that a 
         * valid value has been set in 'outAddress'.
         */
        static const UINT32 WILD_GUESS_HAS_ADDRESS;

        /**
         * Flag that is returned by wildGuessSplitInput() indicating that a 
         * valid value has been set in 'outDevice'.
         */
        static const UINT32 WILD_GUESS_HAS_DEVICE;

        /**
         * Flag that is returned by wildGuessSplitInput() indicating that a 
         * valid value has been set in 'outPrefix', but this prefix was
         * derived from an IPv4 subnet mask rather than from the prefix length
         * itself.
         */
        static const UINT32 WILD_GUESS_HAS_NETMASK;

        /**
         * Flag that is returned by wildGuessSplitInput() indicating that a 
         * valid value has been set in 'outPort'.
         */
        static const UINT32 WILD_GUESS_HAS_PORT;
        
        /**
         * Flag that is returned by wildGuessSplitInput() indicating that a 
         * valid value has been set in 'outPrefix'.
         */
        static const UINT32 WILD_GUESS_HAS_PREFIX_LEN;

        /** The list of adapters. */
        static AdapterList adapters;

        /** A lock protecting the 'adapters' member. */
        static sys::CriticalSection lockAdapters;

        /** 
         * Forbidden Ctor. 
         *
         * @throws UnsupportedOperationException Unconditionally.
         */
        NetworkInformation(void);

       /** 
         * Forbidden Ctor. 
         *
         * @throws UnsupportedOperationException Unconditionally.
         */
        NetworkInformation(const NetworkInformation& rhs);

        /**
         * Dtor. 
         */
        ~NetworkInformation(void);
    };


    ////////////////////////////////////////////////////////////////////////////
    // AssessedMember

    /*
     * vislib::net::NetworkInformation::AssessedMember<T>::AssessedMember
     */
    template<class T>
    NetworkInformation::AssessedMember<T>::AssessedMember(void) 
            : confidence(INVALID) {
        VLSTACKTRACE("AssessedMember::AssessedMember", __FILE__, __LINE__);
    }


    /*
     * vislib::net::NetworkInformation::AssessedMember<T>::~AssessedMember
     */
    template<class T>
    NetworkInformation::AssessedMember<T>::~AssessedMember(void) {
        VLSTACKTRACE("AssessedMember::~AssessedMember", __FILE__, __LINE__);
    }


    /*
     * vislib::net::NetworkInformation::AssessedMember<T>::AssessedMember
     */
    template<class T>
    NetworkInformation::AssessedMember<T>::AssessedMember(
            const T& value, const Confidence confidence) 
            : confidence(confidence), value(value) {
        VLSTACKTRACE("AssessedMember::AssessedMember", __FILE__, __LINE__);
    }


    /*
     * vislib::net::NetworkInformation::AssessedMember<T>::::GetConfidence
     */
    template<class T>
    void NetworkInformation::AssessedMember<T>::GetConfidence(
            Confidence *outConfidence, const char *name) const {
        VLSTACKTRACE("AssessedMember::GetConfidence", __FILE__, __LINE__);
        if (outConfidence != NULL) {
            *outConfidence = this->confidence;
        } else if (this->confidence == INVALID) {
            throw NoConfidenceException(name, __FILE__, __LINE__);
        }
    }


    /*
     * vislib::net::NetworkInformation::Adapter::AssessedMember<T>::operator =
     */
    template<class T>
    NetworkInformation::AssessedMember<T>& 
    NetworkInformation::AssessedMember<T>::operator =(
            const AssessedMember& rhs) {
        VLSTACKTRACE("AssessedMember::operator =", __FILE__, __LINE__);

        if (this != &rhs) {
            this->confidence = rhs.confidence;
            this->value = rhs.value;
        }

        return *this;
    }

    // AssessedMember
    ////////////////////////////////////////////////////////////////////////////


    /*
     * vislib::net::NetworkInformation::findAddress
     */
    template<class A> int NetworkInformation::findAddress(
            const UnicastAddressList& list, const A& addr, const int startIdx) {
        VLSTACKTRACE("NetworkInformation::findAddress", __FILE__, 
            __LINE__);
        ASSERT(startIdx >= 0);

        for (SIZE_T i = static_cast<SIZE_T>(startIdx); i < list.Count(); i++) {
            if (list[i] == addr) {
                return static_cast<int>(i);
            }
        }
        /* Not found. */

        return -1;
    }


    /*
     * NetworkInformation::findSamePrefix
     */
    template<class A> int NetworkInformation::findSamePrefix(
            const UnicastAddressList& list, 
            const A& addr, const ULONG prefixLen, 
            const int startIdx) {
        VLSTACKTRACE("NetworkInformation::findAddress", __FILE__, 
            __LINE__);
        ASSERT(startIdx >= 0);

        try {
            A prefix = addr.GetPrefix(prefixLen);

            for (SIZE_T i = static_cast<SIZE_T>(startIdx); i < list.Count();
                    i++) {
                try {
                    if (list[i].GetAddress().GetPrefix(prefixLen) == prefix) {
                        return static_cast<int>(i);
                    }
                } catch (...) {
                    // Probably tried to get IPv6 prefix from IPv4 address. 
                    // Ignore that as "no match".
                }
            }
            /* Not found. */
        } catch (...) {
            // Illegal prefix length as above.
        }

        return -1;
    }
    
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_NETWORKINFORMATION_H_INCLUDED */
