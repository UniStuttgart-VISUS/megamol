/*
 * InstanceDescription.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_VIEWINSTANCEREQUEST_H_INCLUDED
#define MEGAMOLCORE_VIEWINSTANCEREQUEST_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/InstanceRequest.h"
#include "mmcore/ViewDescription.h"


namespace megamol {
namespace core {

/**
 * Abstract base class of job and view descriptions.
 */
class ViewInstanceRequest : public InstanceRequest {
public:
    /**
     * Ctor.
     */
    ViewInstanceRequest(void);

    /**
     * Copy ctor.
     *
     * @param src The object to clone from
     */
    ViewInstanceRequest(const ViewInstanceRequest& src);

    /**
     * Dtor.
     */
    virtual ~ViewInstanceRequest(void);

    /**
     * Answer the description of the view to be instantiated.
     *
     * @return The description of the view to be instantiated
     */
    inline const ViewDescription* Description(void) const {
        return this->desc;
    }

    /**
     * Sets the description of the view to be instantiated.
     *
     * @param desc The description of the view to be instantiated.
     */
    inline void SetDescription(const ViewDescription* desc) {
        this->desc = desc;
    }

    /**
     * Assignment operator.
     *
     * @param rhs The right hand side operand.
     *
     * @return Reference to 'this'
     */
    ViewInstanceRequest& operator=(const ViewInstanceRequest& rhs);

    /**
     * Test for equality
     *
     * @param rhs The right hand side operand.
     *
     * @return 'true' if 'this' is equal to 'rhs'
     */
    bool operator==(const ViewInstanceRequest& rhs) const;

private:
    /** The view description to be instantiated */
    const ViewDescription* desc;
};

} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_VIEWINSTANCEREQUEST_H_INCLUDED */
