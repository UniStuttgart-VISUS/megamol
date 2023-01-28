/*
 * AbstractCommChannel.h
 *
 * Copyright (C) 2010 by Christoph Müller. Alle Rechte vorbehalten.
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTCOMMCHANNEL_H_INCLUDED
#define VISLIB_ABSTRACTCOMMCHANNEL_H_INCLUDED
#pragma once
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/net/AbstractCommClientChannel.h"
#include "vislib/net/AbstractCommServerChannel.h"

namespace vislib::net {


/**
 * This is the superclass of the VISlib communication channel abstraction
 * layer. This layer is intended to provide a common class-based interface
 * for different network technologies.
 *
 * The AbstractCommChannel represents a bidiectional, full duplex
 * communication channel. Subclasses must implement this behaviour by
 * providing an implementation for all pure virtual methods defined by this
 * interface.
 *
 * Additionally, the interface defines the client connection facilities.
 * Server functionality is defined in the derived AbstractCommServerChannel
 * class.
 *
 * Note for implementors: Subclasses should provide static Create()
 * methods which create objects on the heap that must have been created
 * with C++ new. The Release() method of this class assumes creation
 * with C++ new and releases the object be calling delete once the last
 * reference was released.
 *
 * Rationale: Due to the design-inherent polymorphism of this abstraction
 * layer, we use reference counting for managing the objects. This is
 * because some classes in the layer must return objects on the heap. Users
 * of AbstractCommChannel should use SmartRef for handling the reference
 * counting.
 */
class AbstractCommChannel : public AbstractCommClientChannel, public AbstractCommServerChannel {

public:
    /** Constant for specifying an infinite timeout. */
    static const UINT TIMEOUT_INFINITE;

protected:
    /** Ctor. */
    AbstractCommChannel();

    /** Dtor. */
    ~AbstractCommChannel() override;
};

} // namespace vislib::net

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTCOMMCHANNEL_H_INCLUDED */
