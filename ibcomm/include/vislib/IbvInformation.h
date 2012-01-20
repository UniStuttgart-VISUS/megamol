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

            /** Allow outer class creating instances. */
            friend class ArrayElementDftCtor<Port>;
            friend class IbvInformation;
        };

        /** A list of InfiniBand ports. */
        typedef Array<Port> PortList;

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

            inline NET64 GetNodeGuid(void) const {
                VLSTACKTRACE("Device::GetNodeGuid", __FILE__, __LINE__);
                return this->attributes.NodeGuid;
            }

            StringA GetNodeGuidA(void) const;

            StringW GetNodeGuidW(void) const;

            inline const PortList& GetPorts(void) const {
                VLSTACKTRACE("Device::GetPorts", __FILE__, __LINE__);
                return this->ports;
            }

            inline int GetPortCount(void) const {
                VLSTACKTRACE("Device::GetPortCount", __FILE__, __LINE__);
                return (int) this->attributes.PhysPortCount;
            }

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

        /** A list of InfiniBand devices. */
        typedef Array<Device> DeviceList;

        /**
         * Gets all available InfiniBand devices into 'outDevices'.
         *
         * The array will be erased before new devices are added.
         *
         * @param outDevices Receives the devices.
         *
         * @return The number of devices actually retrieved.
         *
         * @throws vislib::sys::COMException In case of an error.
         */
        static SIZE_T GetDevices(DeviceList& outDevices);

    private:

        /**
         * Get the (cached) WinVerbs root object.
         *
         * @return An instance of IWVProvider.
         *
         * @throws vislib::sys::COMException In case of an error.
         */
        static IWVProvider *getWvProvider(void);

        /** The WinVerbs root object required to get all information. */
        static IWVProvider *wvProvider;

        /** Disallow instances. */
        IbvInformation(void);

        /** Disallow instances. */
        ~IbvInformation(void);
    };
    
} /* end namespace ib */
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_IBVINFORMATION_H_INCLUDED */

