/*
 * ShallowSimpleMessageHeader.h
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_SHALLOWSIMPLEMESSAGEHEADER_H_INCLUDED
#define VISLIB_SHALLOWSIMPLEMESSAGEHEADER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/assert.h"
#include "vislib/net/AbstractSimpleMessageHeader.h"


namespace vislib {
namespace net {


/**
 * The ShallowSimpleMessageHeader allows the interpretation of a memory
 * range as message header data. The actual data are user-provided and the
 * class does not take ownership. The user is responsible for ensuring that
 * the underlying data live as long as the ShallowSimpleMessageHeader lives.
 */
class ShallowSimpleMessageHeader : public AbstractSimpleMessageHeader {

public:
    /**
     * Create a new instance using the message header data designated by
     * 'data'.
     *
     * @param data The message header data. The user must ensure that the data
     *             remain valid as long as this object exists. This pointer must
     *             not be NULL.
     */
    explicit ShallowSimpleMessageHeader(SimpleMessageHeaderData* data);

    /** Dtor. */
    ~ShallowSimpleMessageHeader(void) override;

    /**
     * Provides direct access to the underlying SimpleMessageHeaderData.
     *
     * @return A pointer to the message header data.
     */
    SimpleMessageHeaderData* PeekData(void) override;

    /**
     * Provides direct access to the underlying SimpleMessageHeaderData.
     *
     * @return A pointer to the message header data.
     */
    const SimpleMessageHeaderData* PeekData(void) const override;

    /**
     * Set a new data pointer.
     *
     * The caller remains owner of the memory and must ensure that it lives
     * as long as this object.
     *
     * @param data The message header data. The user must ensure that the data
     *             remain valid as long as this object exists. This pointer must
     *             not be NULL.
     */
    inline void SetData(SimpleMessageHeaderData* data) {
        ASSERT(data != NULL);
        this->data = data;
    }

    /**
     * Assignment operator.
     *
     * @param The right hand side operand.
     *
     * @return *this
     */
    inline ShallowSimpleMessageHeader& operator=(const ShallowSimpleMessageHeader& rhs) {
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
    inline ShallowSimpleMessageHeader& operator=(const AbstractSimpleMessageHeader& rhs) {
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
    inline ShallowSimpleMessageHeader& operator=(const SimpleMessageHeaderData& rhs) {
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
    inline ShallowSimpleMessageHeader& operator=(const SimpleMessageHeaderData* rhs) {
        Super::operator=(rhs);
        return *this;
    }

private:
    /** Superclass typedef. */
    typedef AbstractSimpleMessageHeader Super;

    /**
     * The default ctor creates a new object having NULL as 'data'. This
     * ctor is only intended for vislib-internal use as it leaves illegal
     * instances.
     */
    ShallowSimpleMessageHeader(void);

    /**
     * Pointer to the actual data. The object is not the owner of the memory
     * designated by this pointer.
     */
    SimpleMessageHeaderData* data;

    /** Allow the message class using the default ctor. */
    friend class AbstractSimpleMessage;
};

} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_SHALLOWSIMPLEMESSAGEHEADER_H_INCLUDED */
