/*
 * IbvInformation.h
 *
 * Copyright (C) 2006 - 2012 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_IBVINFORMATION_H_INCLUDED
#define VISLIB_IBVINFORMATION_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/Socket.h"      // Must be first!
#include "vislib/Array.h"
#include "vislib/CriticalSection.h"
#include "vislib/StackTrace.h"

#include "rdma/winverbs.h"


namespace vislib {
namespace net {
namespace ib {


    /**
     * Provides information about the InfiniBand network available on the 
     * machine.
     */
    class IbvInformation {

    public:

#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma region Port
#endif
        /**
         * This class repesents an InfiniBand port.
         */
        class Port {

        public:

            /**
             * Clone 'rhs'.
             *
             * @param rhs The object to be cloned.
             */
            Port(const Port& rhs);

            /** Dtor. */
            ~Port(void);

            /**
             * Answer the native port properties.
             *
             * @return The WV_PORT_ATTRIBUTES of the device.
             */
            inline const WV_PORT_ATTRIBUTES& GetAttributes(void) const {
                VLSTACKTRACE("Port::GetAttributes", __FILE__, __LINE__);
                return this->attributes;
            }

            /**
             * Get the port GUID and prefix GID.
             *
             * @return The port GUID and prefix GID.
             */
            inline const WV_GID& GetGid(void) const {
                VLSTACKTRACE("Port::GetGid", __FILE__, __LINE__);
                return this->gid;
            }

            /**
             * Gets the physical state of the port.
             *
             * You can use GetPhysicalStateA() or GetPhysicalStateW() to 
             * obtain a more comprehensible description of the state.
             *
             * @return A numeric value identifying the current state of the 
             *         port.
             */
            inline UINT8 GetPhysicalState(void) const {
                VLSTACKTRACE("Port::GetPhysicalState", __FILE__, __LINE__);
                return this->attributes.PhysicalState;
            }

            /**
             * Gets a human-readable description of the physical port state.
             *
             * @return A description of the physical port state.
             */
            inline StringA GetPhysicalStateA(void) const {
                VLSTACKTRACE("Port::GetPhysicalStateA", __FILE__, __LINE__);
                return PHYSICAL_STATES[this->attributes.PhysicalState];
            }

            /**
             * Gets a human-readable description of the physical port state.
             *
             * @return A description of the physical port state.
             */
            inline StringW GetPhysicalStateW(void) const {
                VLSTACKTRACE("Port::GetPhysicalStateW", __FILE__, __LINE__);
                return StringW(PHYSICAL_STATES[this->attributes.PhysicalState]);
            }

            /**
             * Get the port GUID.
             *
             * @return The port GUID.
             */
            inline NET64 GetPortGuid(void) const {
                VLSTACKTRACE("Port::GetPortGuid", __FILE__, __LINE__);
                return *(reinterpret_cast<const NET64 *>(&this->gid) + 1);
            }

            /**
             * Get a hex-string representation of the port GUID.
             *
             * @return The port GUID as hex-string.
             */
            StringA GetPortGuidA(void) const;

            /**
             * Get a hex-string representation of the port GUID.
             *
             * @return The port GUID as hex-string.
             */
            StringW GetPortGuidW(void) const;

            /**
             * Gets the current state of the port.
             *
             * @return The state of the port.
             */
            inline WV_PORT_STATE GetState(void) const {
                VLSTACKTRACE("Port::GetState", __FILE__, __LINE__);
                return this->attributes.State;
            }

            /**
             * Get a human-readable description of the port state.
             *
             * @return The state of the port.
             */
            inline StringA GetStateA(void) const {
                VLSTACKTRACE("Port::GetStateA", __FILE__, __LINE__);
                return STATES[this->attributes.State];
            }

            /**
             * Get a human-readable description of the port state.
             *
             * @return The state of the port.
             */             
            inline StringW GetStateW(void) const {
                VLSTACKTRACE("Port::GetStateW", __FILE__, __LINE__);
                return StringW(STATES[this->attributes.State]);
            }

            /**
             * Assignment.
             *
             * @param rhs The right hand side operand.
             *
             * @return *this.
             */
            Port& operator =(const Port& rhs);
            
            /**
             * Test for equality.
             *
             * @param rhs The right hand side operand.
             *
             * @returns true if this object and 'rhs' are equal, 
             *          false otherwise.
             */
            bool operator ==(const Port& rhs) const;

            /**
             * Test for equality.
             *
             * @param rhs The right hand side operand.
             *
             * @returns true if this object and 'rhs' are equal, 
             *          false otherwise.
             */
            inline bool operator !=(const Port& rhs) const {
                VLSTACKTRACE("Port::operator !=", __FILE__, __LINE__);
                return !(*this == rhs);
            }

        private:

            /** 
             * Human-readable descriptions of physical port states. 
             *
             * These strings are borrowed from the source code of "ibstat". They
             * are indexed using the numerical value of the physical port state.
             */
            static const char *PHYSICAL_STATES[];

            /** 
             * Human-readable descriptions of port states. 
             *
             * These strings are borrowed from the source code of "ibstat". They
             * are indexed using the numerical value of the port state 
             * enumeration.
             */
            static const char *STATES[];

            /**
             * Initialise an empty instance.
             *
             * This ctor is required for constructing arrays.
             */
            Port(void);

            /**
             * Create a new instance.
             *
             *
             * @throws vislib::sys::COMException In case of an error.
             */
            Port(IWVDevice *device, const UINT8 port);

            /** Holds the attributes of the device port. */
            WV_PORT_ATTRIBUTES attributes;

            /** 
             * The port GUID (like in umad.cpp, we assume GID 0 contains port 
             * GUID and gid prefix.
             */
            WV_GID gid;

            /** Allow outer class creating instances. */
            friend class ArrayElementDftCtor<Port>;
            friend class IbvInformation;
        };
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma endregion Port
#endif

        /** A list of InfiniBand ports. */
        typedef Array<Port> PortList;

#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma region Device
#endif
        /**
         * This class represents an InfiniBand device.
         */
        class Device {

        public:

            /**
             * Clone 'rhs'.
             *
             * @param rhs The object to be cloned.
             */
            Device(const Device& rhs);

            /** Dtor. */
            ~Device(void);

            /**
             * Answer the native device properties.
             *
             * @return The WV_DEVICE_ATTRIBUTES of the device.
             */
            inline const WV_DEVICE_ATTRIBUTES& GetAttributes(void) const {
                VLSTACKTRACE("Device::GetAttributes", __FILE__, __LINE__);
                return this->attributes;
            }

            /**
             * Get the GUID of the node (device).
             *
             * @return The node GUID.
             */
            inline NET64 GetNodeGuid(void) const {
                VLSTACKTRACE("Device::GetNodeGuid", __FILE__, __LINE__);
                return this->attributes.NodeGuid;
            }

            /**
             * Get a hex-string representation of the node GUID.
             *
             * @return The node GUID.
             */
            StringA GetNodeGuidA(void) const;

            /**
             * Get a hex-string representation of the node GUID.
             *
             * @return The node GUID.
             */
            StringW GetNodeGuidW(void) const;

            /**
             * Gets the descriptor object for the 'idx'th port.
             *
             * @param idx The index of the port to retrieve; must be within
             *            [0, this->GetPortCount[.
             * 
             * @return The descriptor object for the given port.
             *
             * @throws OufOfRangeException If 'idx' does not designate a valid
             *                             port index within 
             *                             [0, this->GetPortCount[.
             */
            const Port& GetPort(const SIZE_T idx) const;

            /**
             * Get the list of port descriptors.
             *
             * @return Lost of ports.
             */
            inline const PortList& GetPorts(void) const {
                VLSTACKTRACE("Device::GetPorts", __FILE__, __LINE__);
                return this->ports;
            }

            /**
             * Get the number of ports the device has.
             *
             * @return The number of ports the device has.
             */
            inline int GetPortCount(void) const {
                VLSTACKTRACE("Device::GetPortCount", __FILE__, __LINE__);
                return (int) this->attributes.PhysPortCount;
            }

            /**
             * Get the system image GUID.
             *
             * @return The system image GUID.
             */
            inline NET64 GetSystemImageGuid(void) const {
                VLSTACKTRACE("Device::GetSystemImageGuid", __FILE__, __LINE__);
                return this->attributes.SystemImageGuid;
            }

            /**
             * Get a hex-string representation of system image GUID.
             *
             * @return The system image GUID.
             */
            StringA GetSystemImageGuidA(void) const;

            /**
             * Get a hex-string representation of system image GUID.
             *
             * @return The system image GUID.
             */
            StringW GetSystemImageGuidW(void) const;

            /**
             * Assignment.
             *
             * @param rhs The right hand side operand.
             *
             * @return *this.
             */
            Device& operator =(const Device& rhs);
            
            /**
             * Test for equality.
             *
             * @param rhs The right hand side operand.
             *
             * @returns true if this object and 'rhs' are equal, 
             *          false otherwise.
             */
            bool operator ==(const Device& rhs) const;

            /**
             * Test for equality.
             *
             * @param rhs The right hand side operand.
             *
             * @returns true if this object and 'rhs' are equal, 
             *          false otherwise.
             */
            inline bool operator !=(const Device& rhs) const {
                VLSTACKTRACE("Device::operator !=", __FILE__, __LINE__);
                return !(*this == rhs);
            }

        private:

            /**
             * Initialise an empty instance.
             *
             * This ctor is required for constructing arrays.
             */
            Device(void);

            /**
             * Create a new instance.
             *
             * @param wvProvider The WinVerbs root object.
             * @param guid       The GUID to retrieve the attributes for.
             *
             * @throws vislib::sys::COMException In case of an error.
             */
            Device(IWVProvider *wvProvider, const NET64& guid);

            /** The attributes describing the device. */
            WV_DEVICE_ATTRIBUTES attributes;

            /** The native WinVerbs device. */
            IWVDevice *device;

            /** Holds the ports. */
            PortList ports;

            /** Allow outer class creating instances. */
            friend class ArrayElementDftCtor<Device>;
            friend class IbvInformation;
        };
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma endregion Device
#endif

        /** A list of InfiniBand devices. */
        typedef Array<Device> DeviceList;

        /**
         * Gets the only instance of the class.
         *
         * @return A reference to the only instance of IbvInformation.
         */
        static IbvInformation& GetInstance(void);

        /**
         * Answer whether the given GID is all-null.
         *
         * @param gid                The GID to be tested.
         * @param ignoreSubnetPrefix If true, ignore the first 64 bits of the 
         *                           GID, which contain the subnet prefix.
         *
         * @return true if the GID is all-null, false otherwise.
         */
        static bool IsNullGid(const WV_GID& gid, 
            const bool ignoreSubnetPrefix = false);

        /**
         * Discards all cached information and immediately re-reads them if
         * requested.
         *
         * This method is thread-safe.
         *
         * @param reread If true, update the cached devices immediately.
         *
         * @throws std::bad_alloc In case of too few memory to store the data.
         * @throws vislib::sys::COMException In case of an error.
         */
        void DiscardCache(const bool reread = false);

        SIZE_T GetDevices(DeviceList& outDevices) const;

    private:

        /** Disallow instances. */
        IbvInformation(void);

        /**
         * Disallow copies.
         *
         * @param rhs The object to be cloned.
         *
         * @throws UnsupportedOperationException Unconditionally.
         */
        IbvInformation(const IbvInformation& rhs);

        /** Disallow instances. */
        ~IbvInformation(void);

        /**
         * Fill the 'devices' cache list if the cache list is empty. If the 
         * cache list 'devices' is not empty, this method will do nothing.
         *
         * This method is NOT thread-safe!
         *
         * @return true If the cache list was filled, false if it was already
         *         full before the method was called.
         *
         * @throws std::bad_alloc In case of too few memory to store the data.
         * @throws vislib::sys::COMException In case of an error.
         */
        bool cacheDevices(void) const;

        /**
         * Disallow assignments.
         *
         * @param rhs The object to be cloned.
         *
         * @return *this.
         *
         * @throws IllegalParamException If (this != &rhs).
         */
        IbvInformation& operator =(const IbvInformation& rhs);

        /** Cache for devices. */
        mutable DeviceList devices;

        /** Lock for cached data. */
        mutable sys::CriticalSection lock;

        /** The WinVerbs root object required to get all information. */
        IWVProvider *wvProvider;
    };
    
} /* end namespace ib */
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_IBVINFORMATION_H_INCLUDED */

