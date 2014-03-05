/*
 * IbvInformation.cpp
 *
 * Copyright (C) 2006 - 2012 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/IbvInformation.h"

#include "vislib/AutoLock.h"
#include "vislib/COMException.h"
#include "vislib/memutils.h"
#include "vislib/IllegalParamException.h"
#include "vislib/OutOfRangeException.h"
#include "vislib/RawStorage.h"
#include "vislib/sysfunctions.h"
#include "vislib/Trace.h"
#include "vislib/UnsupportedOperationException.h"
#include "vislib/utils.h"


#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma region Port
#endif

/*
 * vislib::net::ib::IbvInformation::Port::Port
 */
vislib::net::ib::IbvInformation::Port::Port(const Port& rhs) {
    THE_STACK_TRACE;
    *this = rhs;
}


/*
 * vislib::net::ib::IbvInformation::Port::~Port
 */
vislib::net::ib::IbvInformation::Port::~Port(void) {
    THE_STACK_TRACE;
}


/*
 * vislib::net::ib::IbvInformation::Port::GetPortGuidA
 */
vislib::StringA vislib::net::ib::IbvInformation::Port::GetPortGuidA(
        void) const {
    THE_STACK_TRACE;
    NET64 guid = this->GetPortGuid();
    return BytesToHexStringA(reinterpret_cast<const BYTE *>(&guid), 
        sizeof(guid));
}


/*
 * vislib::net::ib::IbvInformation::Port::GetPortGuidW
 */
vislib::StringW vislib::net::ib::IbvInformation::Port::GetPortGuidW(
        void) const {
    THE_STACK_TRACE;
    NET64 guid = this->GetPortGuid();
    return BytesToHexStringW(reinterpret_cast<const BYTE *>(&guid), 
        sizeof(guid));
}


/*
 * vislib::net::ib::IbvInformation::Port::operator =
 */
vislib::net::ib::IbvInformation::Port& 
vislib::net::ib::IbvInformation::Port::operator =(const Port& rhs) {
    THE_STACK_TRACE;

    if (this != &rhs) {
        ::memcpy(&this->attributes, &rhs.attributes, sizeof(this->attributes));
        this->gid = rhs.gid;
    }

    return *this;
}


/*
 * vislib::net::ib::IbvInformation::Port::operator ==
 */
bool vislib::net::ib::IbvInformation::Port::operator ==(
        const Port& rhs) const {
    THE_STACK_TRACE;

    if (this == &rhs) {
        return true;

    } else {
        // 'gid' is shorter than 'attributes' and should be unique...
        return (::memcmp(&this->gid, &rhs.gid, sizeof(this->gid)) == 0)
            && (::memcmp(&this->attributes, &rhs.attributes, 
            sizeof(this->attributes)) == 0);
    }
}


/*
 * vislib::net::ib::IbvInformation::Port::PHYSICAL_STATES
 */
const char *vislib::net::ib::IbvInformation::Port::PHYSICAL_STATES[] = {
    "No state change",
    "Sleep",
    "Polling",
    "Disabled",
    "PortConfigurationTraining",
    "LinkUp",
    "LinkErrorRecovery",
    "PhyTest"
};


/*
 * vislib::net::ib::IbvInformation::Port::STATES
 */
const char *vislib::net::ib::IbvInformation::Port::STATES[] = {
    "???", 
    "Down", 
    "Initializing",
    "Armed",
    "Active"
};


/*
 * vislib::net::ib::IbvInformation::Port::Port
 */
vislib::net::ib::IbvInformation::Port::Port(void) {
    THE_STACK_TRACE;
    ::ZeroMemory(&this->attributes, sizeof(this->attributes));
}


/*
 * vislib::net::ib::IbvInformation::Port::Port
 */
vislib::net::ib::IbvInformation::Port::Port(IWVDevice *device, 
        const UINT8 port) {
    THE_STACK_TRACE;
    HRESULT hr = S_OK;

    ::ZeroMemory(&this->attributes, sizeof(this->attributes));

    VLTRACE(vislib::Trace::LEVEL_VL_VERBOSE, "Querying port attributes...\n");
    ASSERT(device != NULL);
    if (FAILED(hr = device->QueryPort(port, &this->attributes))) {
        VLTRACE(vislib::Trace::LEVEL_VL_ERROR, "Querying port attributes "
            "failed with error code %d.\n", hr);
        this->~Port();
        throw sys::COMException(hr, __FILE__, __LINE__);
    }

    VLTRACE(vislib::Trace::LEVEL_VL_VERBOSE, "Querying port GUID...\n");
    ASSERT(this->attributes.GidTableLength > 0);
    ASSERT(device != NULL);
    if (FAILED(device->QueryGid(port, 0, &this->gid))) {
        VLTRACE(vislib::Trace::LEVEL_VL_ERROR, "Querying port GUID failed "
            "with error code %d.\n", hr);
        this->~Port();
        throw sys::COMException(hr, __FILE__, __LINE__);
    }
    ASSERT(!IbvInformation::IsNullGid(this->gid));
}
 
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma endregion Port
#endif


#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma region Device
#endif

/*
 * vislib::net::ib::IbvInformation::Device::Device
 */
vislib::net::ib::IbvInformation::Device::Device(const Device& rhs) {
    THE_STACK_TRACE;
    *this = rhs;
}


/*
 * vislib::net::ib::IbvInformation::Device::~Device
 */
vislib::net::ib::IbvInformation::Device::~Device(void) {
    THE_STACK_TRACE;
    VLTRACE(vislib::Trace::LEVEL_VL_VERBOSE, "Releasing IB device object...\n");
    sys::SafeRelease(this->device);
}


/*
 * vislib::net::ib::IbvInformation::Device::GetNodeGuidA
 */
vislib::StringA vislib::net::ib::IbvInformation::Device::GetNodeGuidA(
        void) const {
    THE_STACK_TRACE;
    return BytesToHexStringA(reinterpret_cast<const BYTE *>(
        &this->attributes.NodeGuid), sizeof(this->attributes.NodeGuid));
}


/*
 * vislib::net::ib::IbvInformation::Device::GetNodeGuidW
 */
vislib::StringW vislib::net::ib::IbvInformation::Device::GetNodeGuidW(
        void) const {
    THE_STACK_TRACE;
    return BytesToHexStringW(reinterpret_cast<const BYTE *>(
        &this->attributes.NodeGuid), sizeof(this->attributes.NodeGuid));
}


/*
 * vislib::net::ib::IbvInformation::Device::GetPort
 */
const vislib::net::ib::IbvInformation::Port& 
vislib::net::ib::IbvInformation::Device::GetPort(const SIZE_T idx) const {
    THE_STACK_TRACE;
    if ((idx < 0) || (idx > this->ports.Count() - 1)) {
        throw OutOfRangeException(idx, 0, this->ports.Count() - 1, __FILE__, 
            __LINE__);
    } else {
        return this->ports[idx];
    }
}


/*
 * vislib::net::ib::IbvInformation::Device::GetSystemImageGuidA
 */
vislib::StringA 
vislib::net::ib::IbvInformation::Device::GetSystemImageGuidA(void) const {
    THE_STACK_TRACE;
    return BytesToHexStringA(reinterpret_cast<const BYTE *>(
        &this->attributes.SystemImageGuid), 
        sizeof(this->attributes.SystemImageGuid));
}


/*
 * vislib::StringW 
vislib::net::ib::IbvInformation::Device::GetSystemImageGuidW
 */
vislib::StringW 
vislib::net::ib::IbvInformation::Device::GetSystemImageGuidW(void) const {
    THE_STACK_TRACE;
    return BytesToHexStringW(reinterpret_cast<const BYTE *>(
        &this->attributes.SystemImageGuid), 
        sizeof(this->attributes.SystemImageGuid));
}


/*
 * vislib::net::ib::IbvInformation::Device::operator =
 */
vislib::net::ib::IbvInformation::Device& 
vislib::net::ib::IbvInformation::Device::operator =(const Device& rhs) {
    THE_STACK_TRACE;

    if (this != &rhs) {
        ::memcpy(&this->attributes, &rhs.attributes, sizeof(this->attributes));
        this->device = rhs.device;
        this->device->AddRef();
        this->ports = rhs.ports;
    }

    return *this;
}


/*
 * vislib::net::ib::IbvInformation::Device::operator ==
 */
bool vislib::net::ib::IbvInformation::Device::operator ==(
        const Device& rhs) const {
    THE_STACK_TRACE;

    if (this == &rhs) {
        return true;

    } else {
        return (this->device == rhs.device)
            && (::memcmp(&this->attributes, &rhs.attributes,
            sizeof(this->attributes)) == 0);
    }
}


/*
 * vislib::net::ib::IbvInformation::Device::Device
 */
vislib::net::ib::IbvInformation::Device::Device(void) 
        : device(NULL), ports(NULL) {
    THE_STACK_TRACE;
    ::ZeroMemory(&this->attributes, sizeof(this->attributes));
}


/*
 * vislib::net::ib::IbvInformation::Device::Device
 */
vislib::net::ib::IbvInformation::Device::Device(IWVProvider *wvProvider,
        const NET64& guid) : device(NULL), ports(NULL) {
    THE_STACK_TRACE;
    HRESULT hr = S_OK;

    ::ZeroMemory(&this->attributes, sizeof(this->attributes));

    VLTRACE(vislib::Trace::LEVEL_VL_VERBOSE, "Opening IB device...\n");
    ASSERT(wvProvider != NULL);
    if (FAILED(hr = wvProvider->OpenDevice(guid, &this->device))) {
        VLTRACE(vislib::Trace::LEVEL_VL_ERROR, "Opening device "
            "failed with error code %d.\n", hr);
        this->~Device();
        throw sys::COMException(hr, __FILE__, __LINE__);
    }

    VLTRACE(vislib::Trace::LEVEL_VL_VERBOSE, "Querying device attributes...\n");
    if (FAILED(hr = this->device->Query(&this->attributes))) {
        VLTRACE(vislib::Trace::LEVEL_VL_ERROR, "Querying device attributes "
            "failed with error code %d.\n", hr);
        this->~Device();
        throw sys::COMException(hr, __FILE__, __LINE__);
    }

    for (UINT8 i = 0; i < this->attributes.PhysPortCount; i++) {
        try {
            this->ports.Add(Port(this->device, i + 1));
        } catch (...) {
            this->~Device();
            throw;
        }
    }
}

#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma endregion Device
#endif


/*
 * vislib::net::ib::IbvInformation::GetInstance
 */
vislib::net::ib::IbvInformation& 
vislib::net::ib::IbvInformation::GetInstance(void) {
    THE_STACK_TRACE;
    static IbvInformation instance;
    return instance;
}


/*
 * vislib::net::ib::IbvInformation::IsNullGid
 */
bool vislib::net::ib::IbvInformation::IsNullGid(const WV_GID& gid,
        const bool ignoreSubnetPrefix) {
    THE_STACK_TRACE;
    int cntElements = sizeof(gid.Raw) / sizeof(gid.Raw[0]);

    for (int i = ignoreSubnetPrefix ? 8 : 0; i < cntElements; i++) {
        if (gid.Raw[i] != 0) {
            return false;
        }
    }

    return true;
}


/*
 * vislib::net::ib::IbvInformation::DiscardCache
 */ 
void vislib::net::ib::IbvInformation::DiscardCache(const bool reread) {
    THE_STACK_TRACE;
    sys::AutoLock(this->lock);
    this->devices.Clear();

    if (reread) {
        this->cacheDevices();
    }
}


/*
 * vislib::net::ib::IbvInformation::GetDevices
 */
SIZE_T vislib::net::ib::IbvInformation::GetDevices(
        DeviceList& outDevices) const {
    THE_STACK_TRACE;

    sys::AutoLock(this->lock);
    this->cacheDevices();
    outDevices = this->devices;
    return outDevices.Count();
}


/*
 * vislib::net::ib::IbvInformation::IbvInformation
 */
vislib::net::ib::IbvInformation::IbvInformation(void) : wvProvider(NULL) {
    THE_STACK_TRACE;
    HRESULT hr = S_OK;

    VLTRACE(vislib::Trace::LEVEL_VL_VERBOSE, "Acquiring WinVerbs "
        "provider...\n");
    if (FAILED(hr = ::WvGetObject(IID_IWVProvider, 
            reinterpret_cast<void **>(&this->wvProvider)))) {
        VLTRACE(vislib::Trace::LEVEL_VL_ERROR, "Acquiring WinVerbs provider "
            "failed with error code %d.\n", hr);
        this->~IbvInformation();
        throw sys::COMException(hr, __FILE__, __LINE__);    
    }
}


/*
 * vislib::net::ib::IbvInformation::IbvInformation
 */
vislib::net::ib::IbvInformation::IbvInformation(const IbvInformation& rhs) 
        : wvProvider(NULL) {
    THE_STACK_TRACE;
    throw UnsupportedOperationException("IbvInformation::IbvInformation",
        __FILE__, __LINE__);
}


/*
 * vislib::net::ib::IbvInformation::~IbvInformation
 */
vislib::net::ib::IbvInformation::~IbvInformation(void) {
    THE_STACK_TRACE;
    VLTRACE(vislib::Trace::LEVEL_VL_VERBOSE, "Releasing WinVerbs "
        "provider...\n");
    sys::SafeRelease(this->wvProvider);
}


/*
 * vislib::net::ib::IbvInformation::cacheDevices
 */
bool vislib::net::ib::IbvInformation::cacheDevices(void) const {
    THE_STACK_TRACE;

    RawStorage guids;       // Receives GUIDs of devices.
    SIZE_T size = 0;        // Receives size of GUIDs in bytes.
    HRESULT hr = S_OK;      // Receives WV API results.

    if (this->devices.IsEmpty()) {

        /* Determine number of devices and get their GUIDs. */
        VLTRACE(vislib::Trace::LEVEL_VL_VERBOSE, "Querying number of IB "
            "devices...\n");
        ASSERT(this->wvProvider != NULL);
        if (FAILED(hr = this->wvProvider->QueryDeviceList(NULL, &size))) {
            VLTRACE(vislib::Trace::LEVEL_VL_ERROR, "Querying number of "
                "devices failed with error code %d.\n", hr);
            throw sys::COMException(hr, __FILE__, __LINE__);
        }

        VLTRACE(vislib::Trace::LEVEL_VL_VERBOSE, "Querying device GUIDs...\n");
        guids.AssertSize(size);
        if (FAILED(hr = this->wvProvider->QueryDeviceList(guids.As<NET64>(), 
                &size))) {
            VLTRACE(vislib::Trace::LEVEL_VL_ERROR, "Querying device GUIDs "
                "failed with error code %d.\n", hr);
            throw sys::COMException(hr, __FILE__, __LINE__);
        }

        /* Get the attributes of the devices. */
        size /= sizeof(NET64);
        this->devices.AssertCapacity(size);
        for (SIZE_T i = 0; i < size; i++) {
            NET64 guid = *guids.AsAt<NET64>(i * sizeof(NET64));
            this->devices.Add(Device(wvProvider, guid));
        }

        return true;

    } else {
        return false;

    } /* end if (this->devices.IsEmpty()) */
}


/*
 * vislib::net::ib::IbvInformation::operator =
 */
vislib::net::ib::IbvInformation& 
vislib::net::ib::IbvInformation::operator =(const IbvInformation& rhs) {
    THE_STACK_TRACE;
    if (this != &rhs) {
        throw IllegalParamException("rhs", __FILE__, __LINE__);
    }

    return *this;
}

