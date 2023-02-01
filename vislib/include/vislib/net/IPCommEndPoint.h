/*
 * IPCommEndPoint.h
 *
 * Copyright (C) 2010 by Christoph Müller. Alle Rechte vorbehalten.
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#pragma once
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/StringConverter.h"
#include "vislib/net/AbstractCommEndPoint.h"
#include "vislib/net/IPEndPoint.h"


namespace vislib::net {


/**
 * This class represents a end point address for IP comm channels.
 */
class IPCommEndPoint : public AbstractCommEndPoint {

public:
    /**
     * The IP versions that are supported by IPCommEndPoint.
     */
    enum ProtocolVersion { IPV4 = IPEndPoint::FAMILY_INET, IPV6 = IPEndPoint::FAMILY_INET6 };

    /**
     * Create a new IPCommEndPoint that represents the given
     * IPEndPoint.
     *
     * The caller takes ownership of the object returned. The life time
     * of objects returned by this method is managed by reference counting.
     * The initial reference cound is 1. Additional references can be added
     * by calling AddRef(). The object is destroyed by calling Release() for
     * every reference. It is recommended wrapping all pointers in a
     * SmartRef for automatic reference counting.
     *
     * @param endPoint The end point address.
     *
     * @returns A new IPCommEndPoint that represents the input.
     */
    static SmartRef<AbstractCommEndPoint> Create(const IPEndPoint& endPoint);

    /**
     * Create a new IPCommEndPoint that represents the given
     * address and port.
     *
     * The caller takes ownership of the object returned. The life time
     * of objects returned by this method is managed by reference counting.
     * The initial reference cound is 1. Additional references can be added
     * by calling AddRef(). The object is destroyed by calling Release() for
     * every reference. It is recommended wrapping all pointers in a
     * SmartRef for automatic reference counting.
     *
     * @param ipAddress The address of the end point.
     * @param port      The port of the end point.
     *
     * @returns A new IPCommEndPoint that represents the input.
     */
    static SmartRef<AbstractCommEndPoint> Create(const IPAgnosticAddress& ipAddress, const unsigned short port);

    /**
     * Create a new IPCommEndPoint that represents the given
     * address and port.
     *
     * The caller takes ownership of the object returned. The life time
     * of objects returned by this method is managed by reference counting.
     * The initial reference cound is 1. Additional references can be added
     * by calling AddRef(). The object is destroyed by calling Release() for
     * every reference. It is recommended wrapping all pointers in a
     * SmartRef for automatic reference counting.
     *
     * @param ipAddress The address of the end point.
     * @param port      The port of the end point.
     *
     * @returns A new IPCommEndPoint that represents the input.
     */
    static SmartRef<AbstractCommEndPoint> Create(const IPAddress& ipAddress, const unsigned short port);

    /**
     * Create a new IPCommEndPoint that represents the given
     * address and port.
     *
     * The caller takes ownership of the object returned. The life time
     * of objects returned by this method is managed by reference counting.
     * The initial reference cound is 1. Additional references can be added
     * by calling AddRef(). The object is destroyed by calling Release() for
     * every reference. It is recommended wrapping all pointers in a
     * SmartRef for automatic reference counting.
     *
     * @param ipAddress The address of the end point.
     * @param port      The port of the end point.
     *
     * @returns A new IPCommEndPoint that represents the input.
     */
    static SmartRef<AbstractCommEndPoint> Create(const IPAddress6& ipAddress, const unsigned short port);

    /**
     * Create a new IPCommEndPoint that represents a server end
     * point, i. e. the ANY address.
     *
     * The caller takes ownership of the object returned. The life time
     * of objects returned by this method is managed by reference counting.
     * The initial reference cound is 1. Additional references can be added
     * by calling AddRef(). The object is destroyed by calling Release() for
     * every reference. It is recommended wrapping all pointers in a
     * SmartRef for automatic reference counting.
     *
     * @param protocolVersion The protocol version to be used for the new
     *                        end point.
     * @param port            The port of the end point.
     *
     * @returns A new IPCommEndPoint that represents the input.
     */
    static SmartRef<AbstractCommEndPoint> Create(const ProtocolVersion protocolVersion, const unsigned short port);

    /**
     * Create a new IPCommEndPoint that represents a server end
     * point, i. e. the ANY address.
     *
     * The caller takes ownership of the object returned. The life time
     * of objects returned by this method is managed by reference counting.
     * The initial reference cound is 1. Additional references can be added
     * by calling AddRef(). The object is destroyed by calling Release() for
     * every reference. It is recommended wrapping all pointers in a
     * SmartRef for automatic reference counting.
     *
     * @param addressFamily The protocol version to be used for the new
     *                      end point.
     * @param port          The port of the end point.
     *
     * @returns A new IPCommEndPoint that represents the input.
     *
     * @throws IllegalParamException If 'addressFamily' is not supported.
     */
    inline static SmartRef<AbstractCommEndPoint> Create(
        const IPAgnosticAddress::AddressFamily addressFamily, const unsigned short port) {
        return IPCommEndPoint::Create(IPCommEndPoint::convertAddressFamily(addressFamily), port);
    }

    /**
     * Create a new IPCommEndPoint that represents a server end
     * point, i. e. the ANY address.
     *
     * The caller takes ownership of the object returned. The life time
     * of objects returned by this method is managed by reference counting.
     * The initial reference cound is 1. Additional references can be added
     * by calling AddRef(). The object is destroyed by calling Release() for
     * every reference. It is recommended wrapping all pointers in a
     * SmartRef for automatic reference counting.
     *
     * @param addressFamily The protocol version to be used for the new
     *                      end point.
     * @param port          The port of the end point.
     *
     * @returns A new IPCommEndPoint that represents the input.
     *
     * @throws IllegalParamException If 'addressFamily' is not supported.
     */
    inline static SmartRef<AbstractCommEndPoint> Create(
        const IPEndPoint::AddressFamily addressFamily, const unsigned short port) {
        return IPCommEndPoint::Create(static_cast<IPAgnosticAddress::AddressFamily>(addressFamily), port);
    }

    /**
     * Create a new IPCommEndPoint that represents the given address
     * and port.
     *
     * The caller takes ownership of the object returned. The life time
     * of objects returned by this method is managed by reference counting.
     * The initial reference cound is 1. Additional references can be added
     * by calling AddRef(). The object is destroyed by calling Release() for
     * every reference. It is recommended wrapping all pointers in a
     * SmartRef for automatic reference counting.
     *
     * @param protocolVersion   The protocol version to be used for the new
     *                          end point.
     * @param hostNameOrAddress A host name or IP address in string form.
     * @param port              The port of the end point.
     *
     * @returns A new IPCommEndPoint that represents the input.
     *
     * @throws IllegalParamException If 'hostNameOrAddress' is not a valid
     *                               host name or IP address.
     */
    static SmartRef<AbstractCommEndPoint> Create(
        const ProtocolVersion protocolVersion, const char* hostNameOrAddress, const unsigned short port);

    /**
     * Create a new IPCommEndPoint that represents the given address
     * and port.
     *
     * The caller takes ownership of the object returned. The life time
     * of objects returned by this method is managed by reference counting.
     * The initial reference cound is 1. Additional references can be added
     * by calling AddRef(). The object is destroyed by calling Release() for
     * every reference. It is recommended wrapping all pointers in a
     * SmartRef for automatic reference counting.
     *
     * @param protocolVersion   The protocol version to be used for the new
     *                          end point.
     * @param hostNameOrAddress A host name or IP address in string form.
     * @param port              The port of the end point.
     *
     * @returns A new IPCommEndPoint that represents the input.
     *
     * @throws IllegalParamException If 'hostNameOrAddress' is not a valid
     *                               host name or IP address.
     */
    inline static SmartRef<AbstractCommEndPoint> Create(
        const ProtocolVersion protocolVersion, const wchar_t* hostNameOrAddress, const unsigned short port) {
        return IPCommEndPoint::Create(protocolVersion, W2A(hostNameOrAddress), port);
    }

    /**
     * Create a new IPCommEndPoint that represents the given address
     * and port.
     *
     * The caller takes ownership of the object returned. The life time
     * of objects returned by this method is managed by reference counting.
     * The initial reference cound is 1. Additional references can be added
     * by calling AddRef(). The object is destroyed by calling Release() for
     * every reference. It is recommended wrapping all pointers in a
     * SmartRef for automatic reference counting.
     *
     * @param addressFamily     The protocol version to be used for the new
     *                          end point.
     * @param hostNameOrAddress A host name or IP address in string form.
     * @param port              The port of the end point.
     *
     * @returns A new IPCommEndPoint that represents the input.
     *
     * @throws IllegalParamException If 'hostNameOrAddress' is not a valid
     *                               host name or IP address.
     * @throws IllegalParamException If 'addressFamily' is not supported.
     */
    inline static SmartRef<AbstractCommEndPoint> Create(const IPAgnosticAddress::AddressFamily addressFamily,
        const char* hostNameOrAddress, const unsigned short port) {
        return IPCommEndPoint::Create(IPCommEndPoint::convertAddressFamily(addressFamily), hostNameOrAddress, port);
    }

    /**
     * Create a new IPCommEndPoint that represents the given address
     * and port.
     *
     * The caller takes ownership of the object returned. The life time
     * of objects returned by this method is managed by reference counting.
     * The initial reference cound is 1. Additional references can be added
     * by calling AddRef(). The object is destroyed by calling Release() for
     * every reference. It is recommended wrapping all pointers in a
     * SmartRef for automatic reference counting.
     *
     * @param addressFamily     The protocol version to be used for the new
     *                          end point.
     * @param hostNameOrAddress A host name or IP address in string form.
     * @param port              The port of the end point.
     *
     * @returns A new IPCommEndPoint that represents the input.
     *
     * @throws IllegalParamException If 'hostNameOrAddress' is not a valid
     *                               host name or IP address.
     * @throws IllegalParamException If 'addressFamily' is not supported.
     */
    inline static SmartRef<AbstractCommEndPoint> Create(const IPAgnosticAddress::AddressFamily addressFamily,
        const wchar_t* hostNameOrAddress, const unsigned short port) {
        return IPCommEndPoint::Create(addressFamily, W2A(hostNameOrAddress), port);
    }

    /**
     * Create a new IPCommEndPoint that represents the given address
     * and port.
     *
     * The caller takes ownership of the object returned. The life time
     * of objects returned by this method is managed by reference counting.
     * The initial reference cound is 1. Additional references can be added
     * by calling AddRef(). The object is destroyed by calling Release() for
     * every reference. It is recommended wrapping all pointers in a
     * SmartRef for automatic reference counting.
     *
     * @param addressFamily     The protocol version to be used for the new
     *                          end point.
     * @param hostNameOrAddress A host name or IP address in string form.
     * @param port              The port of the end point.
     *
     * @returns A new IPCommEndPoint that represents the input.
     *
     * @throws IllegalParamException If 'hostNameOrAddress' is not a valid
     *                               host name or IP address.
     * @throws IllegalParamException If 'addressFamily' is not supported.
     */
    inline static SmartRef<AbstractCommEndPoint> Create(
        const IPEndPoint::AddressFamily addressFamily, const char* hostNameOrAddress, const unsigned short port) {
        return IPCommEndPoint::Create(
            static_cast<IPAgnosticAddress::AddressFamily>(addressFamily), hostNameOrAddress, port);
    }

    /**
     * Create a new IPCommEndPoint that represents the given address
     * and port.
     *
     * The caller takes ownership of the object returned. The life time
     * of objects returned by this method is managed by reference counting.
     * The initial reference cound is 1. Additional references can be added
     * by calling AddRef(). The object is destroyed by calling Release() for
     * every reference. It is recommended wrapping all pointers in a
     * SmartRef for automatic reference counting.
     *
     * @param addressFamily     The protocol version to be used for the new
     *                          end point.
     * @param hostNameOrAddress A host name or IP address in string form.
     * @param port              The port of the end point.
     *
     * @returns A new IPCommEndPoint that represents the input.
     *
     * @throws IllegalParamException If 'hostNameOrAddress' is not a valid
     *                               host name or IP address.
     * @throws IllegalParamException If 'addressFamily' is not supported.
     */
    inline static SmartRef<AbstractCommEndPoint> Create(
        const IPEndPoint::AddressFamily addressFamily, const wchar_t* hostNameOrAddress, const unsigned short port) {
        return IPCommEndPoint::Create(addressFamily, W2A(hostNameOrAddress), port);
    }

    /**
     * Create a new IPCommEndPoint that represents the given address
     * string. The address string is expected to be in <address>:<port>
     * format for all supported protocol versions.
     *
     * The caller takes ownership of the object returned. The life time
     * of objects returned by this method is managed by reference counting.
     * The initial reference cound is 1. Additional references can be added
     * by calling AddRef(). The object is destroyed by calling Release() for
     * every reference. It is recommended wrapping all pointers in a
     * SmartRef for automatic reference counting.
     *
     * @param protocolVersion The protocol version to be used for the new
     *                        end point.
     * @param str             The string representation of the address.
     *
     * @returns A new IPCommEndPoint that represents the input.
     *
     * @throws IllegalParamException If 'str' is not a valid end point
     *                               address. "Valid" also means that the
     *                               string representation is compatible
     *                               with the given protocol version.
     */
    inline static SmartRef<AbstractCommEndPoint> Create(const ProtocolVersion protocolVersion, const char* str) {
        return IPCommEndPoint::Create(static_cast<IPAgnosticAddress::AddressFamily>(protocolVersion), str);
    }

    /**
     * Create a new IPCommEndPoint that represents the given address
     * string. The address string is expected to be in <address>:<port>
     * format for all supported protocol versions.
     *
     * The caller takes ownership of the object returned. The life time
     * of objects returned by this method is managed by reference counting.
     * The initial reference cound is 1. Additional references can be added
     * by calling AddRef(). The object is destroyed by calling Release() for
     * every reference. It is recommended wrapping all pointers in a
     * SmartRef for automatic reference counting.
     *
     * @param protocolVersion The protocol version to be used for the new
     *                        end point.
     * @param str             The string representation of the address.
     *
     * @returns A new IPCommEndPoint that represents the input.
     *
     * @throws IllegalParamException If 'str' is not a valid end point
     *                               address. "Valid" also means that the
     *                               string representation is compatible
     *                               with the given protocol version.
     */
    inline static SmartRef<AbstractCommEndPoint> Create(const ProtocolVersion protocolVersion, const wchar_t* str) {
        return IPCommEndPoint::Create(protocolVersion, W2A(str));
    }

    /**
     * Create a new IPCommEndPoint that represents the given address
     * string. The address string is expected to be in <address>:<port>
     * format for all supported protocol versions.
     *
     * The caller takes ownership of the object returned. The life time
     * of objects returned by this method is managed by reference counting.
     * The initial reference cound is 1. Additional references can be added
     * by calling AddRef(). The object is destroyed by calling Release() for
     * every reference. It is recommended wrapping all pointers in a
     * SmartRef for automatic reference counting.
     *
     * @param addressFamily The protocol version to be used for the new
     *                      end point.
     * @param str           The string representation of the address.
     *
     * @returns A new IPCommEndPoint that represents the input.
     *
     * @throws IllegalParamException If 'str' is not a valid end point
     *                               address. "Valid" also means that the
     *                               string representation is compatible
     *                               with the given protocol version.
     * @throws IllegalParamException If 'addressFamily' is not supported.
     */
    static SmartRef<AbstractCommEndPoint> Create(const IPAgnosticAddress::AddressFamily addressFamily, const char* str);

    /**
     * Create a new IPCommEndPoint that represents the given address
     * string. The address string is expected to be in <address>:<port>
     * format for all supported protocol versions.
     *
     * The caller takes ownership of the object returned. The life time
     * of objects returned by this method is managed by reference counting.
     * The initial reference cound is 1. Additional references can be added
     * by calling AddRef(). The object is destroyed by calling Release() for
     * every reference. It is recommended wrapping all pointers in a
     * SmartRef for automatic reference counting.
     *
     * @param addressFamily The protocol version to be used for the new
     *                      end point.
     * @param str           The string representation of the address.
     *
     * @returns A new IPCommEndPoint that represents the input.
     *
     * @throws IllegalParamException If 'str' is not a valid end point
     *                               address. "Valid" also means that the
     *                               string representation is compatible
     *                               with the given protocol version.
     * @throws IllegalParamException If 'addressFamily' is not supported.
     */
    inline static SmartRef<AbstractCommEndPoint> Create(
        const IPAgnosticAddress::AddressFamily addressFamily, const wchar_t* str) {
        return IPCommEndPoint::Create(addressFamily, W2A(str));
    }

    /**
     * Create a new IPCommEndPoint that represents the given address
     * string. The address string is expected to be in <address>:<port>
     * format for all supported protocol versions.
     *
     * The caller takes ownership of the object returned. The life time
     * of objects returned by this method is managed by reference counting.
     * The initial reference cound is 1. Additional references can be added
     * by calling AddRef(). The object is destroyed by calling Release() for
     * every reference. It is recommended wrapping all pointers in a
     * SmartRef for automatic reference counting.
     *
     * @param addressFamily The protocol version to be used for the new
     *                      end point.
     * @param str           The string representation of the address.
     *
     * @returns A new IPCommEndPoint that represents the input.
     *
     * @throws IllegalParamException If 'str' is not a valid end point
     *                               address. "Valid" also means that the
     *                               string representation is compatible
     *                               with the given protocol version.
     * @throws IllegalParamException If 'addressFamily' is not supported.
     */
    inline static SmartRef<AbstractCommEndPoint> Create(
        const IPEndPoint::AddressFamily addressFamily, const char* str) {
        return IPCommEndPoint::Create(static_cast<IPAgnosticAddress::AddressFamily>(addressFamily), str);
    }

    /**
     * Create a new IPCommEndPoint that represents the given address
     * string. The address string is expected to be in <address>:<port>
     * format for all supported protocol versions.
     *
     * The caller takes ownership of the object returned. The life time
     * of objects returned by this method is managed by reference counting.
     * The initial reference cound is 1. Additional references can be added
     * by calling AddRef(). The object is destroyed by calling Release() for
     * every reference. It is recommended wrapping all pointers in a
     * SmartRef for automatic reference counting.
     *
     * @param addressFamily The protocol version to be used for the new
     *                      end point.
     * @param str           The string representation of the address.
     *
     * @returns A new IPCommEndPoint that represents the input.
     *
     * @throws IllegalParamException If 'str' is not a valid end point
     *                               address. "Valid" also means that the
     *                               string representation is compatible
     *                               with the given protocol version.
     * @throws IllegalParamException If 'addressFamily' is not supported.
     */
    inline static SmartRef<AbstractCommEndPoint> Create(
        const IPEndPoint::AddressFamily addressFamily, const wchar_t* str) {
        return IPCommEndPoint::Create(addressFamily, W2A(str));
    }

    /**
     * Create a new IPCommEndPoint that represents the given address
     * string. This method is equivalent to parsing the end point address
     * from a string using the Parse() method. The same restrictions for the
     * input apply and the behaviour regarding the protocol versions is the
     * same.
     *
     * The caller takes ownership of the object returned. The life time
     * of objects returned by this method is managed by reference counting.
     * The initial reference cound is 1. Additional references can be added
     * by calling AddRef(). The object is destroyed by calling Release() for
     * every reference. It is recommended wrapping all pointers in a
     * SmartRef for automatic reference counting.
     *
     * @param str The string representation of the address.
     *
     * @returns A new IPCommEndPoint that represents the input.
     *
     * @throws IllegalParamException If 'str' is not a valid end point
     *                               address.
     */
    static SmartRef<AbstractCommEndPoint> Create(const char* str);

    /**
     * Create a new IPCommEndPoint that represents the given address
     * string. This method is equivalent to parsing the end point address
     * from a string using the Parse() method. The same restrictions for the
     * input apply and the behaviour regarding the protocol versions is the
     * same.
     *
     * The caller takes ownership of the object returned. The life time
     * of objects returned by this method is managed by reference counting.
     * The initial reference cound is 1. Additional references can be added
     * by calling AddRef(). The object is destroyed by calling Release() for
     * every reference. It is recommended wrapping all pointers in a
     * SmartRef for automatic reference counting.
     *
     * @param str The string representation of the address.
     *
     * @returns A new IPCommEndPoint that represents the input.
     *
     * @throws IllegalParamException If 'str' is not a valid end point
     *                               address.
     */
    static SmartRef<AbstractCommEndPoint> Create(const wchar_t* str);

    /**
     * Create a new IPCommEndPoint from an OS IP-agnostic address structure.
     *
     * @param address The address storage.
     */
    static SmartRef<AbstractCommEndPoint> Create(const struct sockaddr_storage& address);

    /**
     * Create a new IPCommEndPoint from an OS IPv4 structure.
     *
     * @param address The IPv4 address to set.
     */
    static SmartRef<AbstractCommEndPoint> Create(const struct sockaddr_in& address);

    /**
     * Create a new IPCommEndPoint from an OS IPv6 structure.
     *
     * @param address The IPv6 address to set.
     */
    static SmartRef<AbstractCommEndPoint> Create(const struct sockaddr_in6& address);


    /**
     * Get the IP address part of the end point address.
     *
     * @return The IP address part of the end point address.
     */
    inline IPAgnosticAddress GetIPAddress() const {
        return this->endPoint.GetIPAddress();
    }

    /**
     * Get the port of the end point address.
     *
     * @return The port of the end point address.
     */
    inline unsigned short GetPort() const {
        return this->endPoint.GetPort();
    }

    /**
     * Answer the protocol version of the end point address.
     *
     * @return The protocol version of the end point address.
     */
    inline ProtocolVersion GetProtocolVersion() const {
        return static_cast<ProtocolVersion>(this->endPoint.GetAddressFamily());
    }

    /**
     * Parses a string as a end point address and sets the current
     * object to this address. An exception is thrown in case it was
     * not possible to parse the input string.
     *
     * The address string is expected to be in <address>:<port>
     * format for all supported protocol versions. Depending on the format
     * of the address, the protocol version is chosen. Please note that
     * IPv6 is preferred over IPv4.
     *
     * @param str A string representation of an end point address.
     *
     * @throws vislib::Exception Or derived in case that 'str' could not
     *                           be parsed as an end point address.
     */
    void Parse(const StringA& str) override;

    /**
     * Parses a string as a end point address and sets the current
     * object to this address. An exception is thrown in case it was
     * not possible to parse the input string.
     *
     * The address string is expected to be in <address>:<port>
     * format for all supported protocol versions.
     *
     * @param str                      A string representation of an end
     *                                 point address.
     * @param preferredProtocolVersion The preferred protocol version if
     *                                 more than one is possible.
     *
     * @throws vislib::Exception Or derived in case that 'str' could not
     *                           be parsed as an end point address.
     */
    virtual void Parse(const StringA& str, const ProtocolVersion preferredProtocolVersion);

    /**
     * Parses a string as a end point address and sets the current
     * object to this address. An exception is thrown in case it was
     * not possible to parse the input string.
     *
     * The address string is expected to be in <address>:<port>
     * format for all supported protocol versions. Depending on the format
     * of the address, the protocol version is chosen. Please note that
     * IPv6 is preferred over IPv4.
     *
     * @param str A string representation of an end point address.
     *
     * @throws vislib::Exception Or derived in case that 'str' could not
     *                           be parsed as an end point address.
     */
    void Parse(const StringW& str) override;

    /**
     * Parses a string as a end point address and sets the current
     * object to this address. An exception is thrown in case it was
     * not possible to parse the input string.
     *
     * The address string is expected to be in <address>:<port>
     * format for all supported protocol versions.
     *
     * @param str                      A string representation of an end
     *                                 point address.
     * @param preferredProtocolVersion The preferred protocol version if
     *                                 more than one is possible.
     *
     * @throws vislib::Exception Or derived in case that 'str' could not
     *                           be parsed as an end point address.
     */
    virtual void Parse(const StringW& str, const ProtocolVersion preferredProtocolVersion);

    /**
     * Set a new end point address.
     *
     * @param endPoint The new end point address.
     */
    inline void SetEndPoint(const IPEndPoint& endPoint) {
        this->endPoint = endPoint;
    }

    /**
     * Set a new IP address.
     *
     * @param ipAddress The new IP address.
     */
    inline void SetIPAddress(const IPAgnosticAddress& ipAddress) {
        this->endPoint.SetIPAddress(ipAddress);
    }

    /**
     * Set a new IP address.
     *
     * @param ipAddress The new IP address.
     */
    inline void SetIPAddress(const IPAddress& ipAddress) {
        this->endPoint.SetIPAddress(ipAddress);
    }

    /**
     * Set a new IP address.
     *
     * @param ipAddress The new IP address.
     */
    inline void SetIPAddress(const IPAddress6& ipAddress) {
        this->endPoint.SetIPAddress(ipAddress);
    }

    /**
     * Set a new port.
     *
     * @param ipAddress The new port.
     */
    inline void SetPort(const unsigned short port) {
        this->endPoint.SetPort(port);
    }

    /**
     * Answer a string representation of the address.
     *
     * @return A string representation of the address.
     */
    StringA ToStringA() const override;

    /**
     * Answer a string representation of the address.
     *
     * @return A string representation of the address.
     */
    StringW ToStringW() const override;

    /**
     * Check for equality.
     *
     * @param rhs The right hand side operand.
     *
     * @return true if this object and 'rhs' are equal, false otherwise.
     */
    bool operator==(const AbstractCommEndPoint& rhs) const override;

    /**
     * Access the underlying IPEndPoint.
     *
     * @return Reference to the underlying IPEndPoint.
     */
    inline operator const IPEndPoint&() const {
        return this->endPoint;
    }

    /**
     * Access the underlying IPEndPoint.
     *
     * @return Reference to the underlying IPEndPoint.
     */
    inline operator IPEndPoint&() {
        return this->endPoint;
    }

private:
    /** Superclass typedef. */
    typedef AbstractCommEndPoint Super;

    /**
     * Checks and converts, if valid, a address family to a protocol
     * version.
     *
     * @param The address family to be converted.
     *
     * @return The protocol version that represents the given address
     *         family.
     *
     * @throws IllegalParamException If 'addressFamily' is not supported.
     */
    static ProtocolVersion convertAddressFamily(const IPAgnosticAddress::AddressFamily addressFamily);

    /**
     * Ctor.
     *
     * @param endPoint The end point to be wrapped.
     */
    IPCommEndPoint(const IPEndPoint& endPoint);

    /** Dtor. */
    ~IPCommEndPoint() override;

    /** The actual IP end point address wrapped by this object. */
    IPEndPoint endPoint;
};

} // namespace vislib::net

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
