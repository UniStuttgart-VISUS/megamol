/*
 * IbvInformation.cpp
 *
 * Copyright (C) 2006 - 2012 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/IbvInformation.h"

#include "vislib/COMException.h"
#include "vislib/memutils.h"
#include "vislib/OutOfRangeException.h"
#include "vislib/RawStorage.h"
#include "vislib/sysfunctions.h"
#include "vislib/Trace.h"
#include "vislib/utils.h"


////////////////////////////////////////////////////////////////////////////////
// BEGIN NESTED CLASS Port

/*
 * vislib::net::ib::IbvInformation::Port::Port
 */
vislib::net::ib::IbvInformation::Port::Port(const Port& rhs) {
    VLSTACKTRACE("Port::Port", __FILE__, __LINE__);
    *this = rhs;
}


/*
 * vislib::net::ib::IbvInformation::Port::~Port
 */
vislib::net::ib::IbvInformation::Port::~Port(void) {
    VLSTACKTRACE("Port::~Port", __FILE__, __LINE__);
}


/*
 * vislib::net::ib::IbvInformation::Port::operator =
 */
vislib::net::ib::IbvInformation::Port& 
vislib::net::ib::IbvInformation::Port::operator =(const Port& rhs) {
    VLSTACKTRACE("Port::operator =", __FILE__, __LINE__);

    if (this != &rhs) {
        ::memcpy(&this->attributes, &rhs.attributes, sizeof(this->attributes));
    }

    return *this;
}


/*
 * vislib::net::ib::IbvInformation::Port::operator ==
 */
bool vislib::net::ib::IbvInformation::Port::operator ==(
        const Port& rhs) const {
    VLSTACKTRACE("Port::operator =", __FILE__, __LINE__);

    if (this == &rhs) {
        return true;

    } else {
        return (::memcmp(&this->attributes, &rhs.attributes, 
            sizeof(this->attributes)) == 0);
    }
}


/*
 * vislib::net::ib::IbvInformation::Port::Port
 */
vislib::net::ib::IbvInformation::Port::Port(void) {
    VLSTACKTRACE("Port::Port", __FILE__, __LINE__);
    ::ZeroMemory(&this->attributes, sizeof(this->attributes));
}


/*
 * vislib::net::ib::IbvInformation::Port::Port
 */
vislib::net::ib::IbvInformation::Port::Port(IWVDevice *device, 
        const UINT8 port) {
    VLSTACKTRACE("Port::Port", __FILE__, __LINE__);
    HRESULT hr = S_OK;

    ::ZeroMemory(&this->attributes, sizeof(this->attributes));

    ASSERT(device != NULL);
    if (FAILED(hr = device->QueryPort(port, &this->attributes))) {
        VLTRACE(vislib::Trace::LEVEL_VL_ERROR, "Querying port attributes"
            "failed with error code %d.\n", hr);
        this->~Port();
        throw sys::COMException(hr, __FILE__, __LINE__);
    }

    //device->QueryGid(
}
            
// BEGIN NESTED CLASS Port
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// BEGIN NESTED CLASS Device

/*
 * vislib::net::ib::IbvInformation::Device::Device
 */
vislib::net::ib::IbvInformation::Device::Device(const Device& rhs) {
    VLSTACKTRACE("Device::Device", __FILE__, __LINE__);
    *this = rhs;
}


/*
 * vislib::net::ib::IbvInformation::Device::~Device
 */
vislib::net::ib::IbvInformation::Device::~Device(void) {
    VLSTACKTRACE("Device::~Device", __FILE__, __LINE__);
    sys::SafeRelease(this->device);
}


/*
 * vislib::net::ib::IbvInformation::Device::GetNodeGuidA
 */
vislib::StringA vislib::net::ib::IbvInformation::Device::GetNodeGuidA(
        void) const {
    VLSTACKTRACE("Device::GetNodeGuidA", __FILE__, __LINE__);
    return BytesToHexStringA(reinterpret_cast<const BYTE *>(
        &this->attributes.NodeGuid), sizeof(this->attributes.NodeGuid));
}


/*
 * vislib::net::ib::IbvInformation::Device::GetNodeGuidW
 */
vislib::StringW vislib::net::ib::IbvInformation::Device::GetNodeGuidW(
        void) const {
    VLSTACKTRACE("Device::GetNodeGuidW", __FILE__, __LINE__);
    return BytesToHexStringW(reinterpret_cast<const BYTE *>(
        &this->attributes.NodeGuid), sizeof(this->attributes.NodeGuid));
}


/*
 * vislib::net::ib::IbvInformation::Device::operator =
 */
vislib::net::ib::IbvInformation::Device& 
vislib::net::ib::IbvInformation::Device::operator =(const Device& rhs) {
    VLSTACKTRACE("Device::operator =", __FILE__, __LINE__);

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
    VLSTACKTRACE("Device::operator =", __FILE__, __LINE__);

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
    VLSTACKTRACE("Device::Device", __FILE__, __LINE__);
    ::ZeroMemory(&this->attributes, sizeof(this->attributes));
}


/*
 * vislib::net::ib::IbvInformation::Device::Device
 */
vislib::net::ib::IbvInformation::Device::Device(IWVProvider *wvProvider,
        const NET64& guid) : device(NULL), ports(NULL) {
    VLSTACKTRACE("Device::Device", __FILE__, __LINE__);
    HRESULT hr = S_OK;

    ::ZeroMemory(&this->attributes, sizeof(this->attributes));

    ASSERT(wvProvider != NULL);
    if (FAILED(hr = wvProvider->OpenDevice(guid, &this->device))) {
        VLTRACE(vislib::Trace::LEVEL_VL_ERROR, "Opening device "
            "failed with error code %d.\n", hr);
        this->~Device();
        throw sys::COMException(hr, __FILE__, __LINE__);
    }

    if (FAILED(hr = this->device->Query(&this->attributes))) {
        VLTRACE(vislib::Trace::LEVEL_VL_ERROR, "Querying device attributes "
            "failed with error code %d.\n", hr);
        this->~Device();
        throw sys::COMException(hr, __FILE__, __LINE__);
    }

    //this->ports = new WV_PORT_ATTRIBUTES[this->attributes.PhysPortCount];
    for (UINT8 i = 0; i < this->attributes.PhysPortCount; i++) {
        try {
            this->ports.Add(Port(this->device, i));
        } catch (...) {
            this->~Device();
            throw;
        }
    }
}

// END NESTED CLASS Device
////////////////////////////////////////////////////////////////////////////////


/*
 * vislib::net::ib::IbvInformation::GetDevices
 */
SIZE_T vislib::net::ib::IbvInformation::GetDevices(DeviceList& outDevices) {
    VLSTACKTRACE("IbvInformation::getWvProvider", __FILE__, __LINE__);

    RawStorage guids;
    SIZE_T retval = 0;
    SIZE_T size = 0;
    HRESULT hr = S_OK;
    IWVProvider *wvProvider = IbvInformation::getWvProvider();

    /* Clear output. */
    outDevices.Clear();

    /* Determine number of devices and get their GUIDs. */
    if (FAILED(hr = wvProvider->QueryDeviceList(NULL, &size))) {
        VLTRACE(vislib::Trace::LEVEL_VL_ERROR, "Querying number of "
            "devices failed with error code %d.\n", hr);
        throw sys::COMException(hr, __FILE__, __LINE__);
    }

    guids.AssertSize(size);
    if (FAILED(hr = wvProvider->QueryDeviceList(guids.As<NET64>(), &size))) {
        VLTRACE(vislib::Trace::LEVEL_VL_ERROR, "Querying device GUIDs "
            "failed with error code %d.\n", hr);
        throw sys::COMException(hr, __FILE__, __LINE__);
    }

    /* Get the attributes of the devices. */
    retval = size / sizeof(NET64);
    outDevices.AssertCapacity(retval);
    for (SIZE_T i = 0; i < retval; i++) {
        NET64 guid = *guids.AsAt<NET64>(i * sizeof(NET64));
        outDevices.Add(Device(wvProvider, guid));
    }

    return retval;
}


/*
 * vislib::net::ib::IbvInformation::getWvProvider
 */
IWVProvider *vislib::net::ib::IbvInformation::getWvProvider(void) {
    VLSTACKTRACE("IbvInformation::getWvProvider", __FILE__, __LINE__);
    HRESULT hr = S_OK;

    if (IbvInformation::wvProvider == NULL) {
        VLTRACE(vislib::Trace::LEVEL_VL_VERBOSE, "Acquiring WinVerbs "
            "provider...\n");
        if (FAILED(hr = ::WvGetObject(IID_IWVProvider, 
                reinterpret_cast<void **>(&IbvInformation::wvProvider)))) {
            VLTRACE(vislib::Trace::LEVEL_VL_ERROR, "Acquiring WinVerbs "
                "provider failed with error code %d.\n", hr);
            throw sys::COMException(hr, __FILE__, __LINE__);    
        }
    }

    ASSERT(IbvInformation::wvProvider != NULL);
    return IbvInformation::wvProvider;
}


/*
 * vislib::net::ib::IbvInformation::wvProvider 
 */
IWVProvider *vislib::net::ib::IbvInformation::wvProvider = NULL;


/*
 * vislib::net::ib::IbvInformation::IbvInformation
 */
vislib::net::ib::IbvInformation::IbvInformation(void) {
    VLSTACKTRACE("IbvInformation::IbvInformation", __FILE__, __LINE__);
}


/*
 * vislib::net::ib::IbvInformation::~IbvInformation
 */
vislib::net::ib::IbvInformation::~IbvInformation(void) {
    VLSTACKTRACE("IbvInformation::~IbvInformation", __FILE__, __LINE__);
}
