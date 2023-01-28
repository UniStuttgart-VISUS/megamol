/*
 * SimpleMessageHeader.h
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_SIMPLEMESSAGEHEADER_H_INCLUDED
#define VISLIB_SIMPLEMESSAGEHEADER_H_INCLUDED
#pragma once
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/net/AbstractSimpleMessageHeader.h"


namespace vislib::net {


/**
 * This class represents a message header consisting of
 * SimpleMessageHeaderData.
 */
class SimpleMessageHeader : public AbstractSimpleMessageHeader {

public:
    /** Ctor. */
    SimpleMessageHeader();

    /**
     * Clone 'rhs'.
     *
     * @param rhs The object to be cloned.
     */
    SimpleMessageHeader(const SimpleMessageHeader& rhs);

    /**
     * Clone 'rhs'.
     *
     * @param rhs The object to be cloned.
     */
    SimpleMessageHeader(const AbstractSimpleMessageHeader& rhs);

    /**
     * Assign the given header data to this message header.
     *
     * @param data The message header data.
     */
    SimpleMessageHeader(const SimpleMessageHeaderData& data);

    /**
     * Assign the given header data to this message header.
     *
     * @param data Pointer to the message header data. This must not be
     *             NULL.
     */
    explicit SimpleMessageHeader(const SimpleMessageHeaderData* data);

    /** Dtor. */
    ~SimpleMessageHeader() override;

    /**
     * Provides direct access to the underlying SimpleMessageHeaderData.
     *
     * @return A pointer to the message header data.
     */
    SimpleMessageHeaderData* PeekData() override;

    /**
     * Provides direct access to the underlying SimpleMessageHeaderData.
     *
     * @return A pointer to the message header data.
     */
    const SimpleMessageHeaderData* PeekData() const override;

    /**
     * Assignment operator.
     *
     * @param The right hand side operand.
     *
     * @return *this
     */
    inline SimpleMessageHeader& operator=(const SimpleMessageHeader& rhs) {
        Super::operator=(rhs);
        return *this;
    }

    /**
     * Assignment operator.
     *
     * @param The right hand side operand.
     *
     * @return *this
     */
    inline SimpleMessageHeader& operator=(const AbstractSimpleMessageHeader& rhs) {
        Super::operator=(rhs);
        return *this;
    }

    /**
     * Assignment operator.
     *
     * @param The right hand side operand.
     *
     * @return *this
     */
    inline SimpleMessageHeader& operator=(const SimpleMessageHeaderData& rhs) {
        Super::operator=(rhs);
        return *this;
    }

    /**
     * Assignment operator.
     *
     * @param The right hand side operand. This must not be NULL.
     *
     * @return *this
     */
    inline SimpleMessageHeader& operator=(const SimpleMessageHeaderData* rhs) {
        Super::operator=(rhs);
        return *this;
    }

private:
    /** Superclass typedef. */
    typedef AbstractSimpleMessageHeader Super;

    /** The actual header data. */
    SimpleMessageHeaderData data;
};


} // namespace vislib::net

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_SIMPLEMESSAGEHEADER_H_INCLUDED */
