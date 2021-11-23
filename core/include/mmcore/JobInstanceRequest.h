/*
 * JobInstanceRequest.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_JOBINSTANCEREQUEST_H_INCLUDED
#define MEGAMOLCORE_JOBINSTANCEREQUEST_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/InstanceRequest.h"
#include "mmcore/JobDescription.h"


namespace megamol {
namespace core {

/**
 * Abstract base class of job and view descriptions.
 */
class JobInstanceRequest : public InstanceRequest {
public:
    /**
     * Ctor.
     */
    JobInstanceRequest(void);

    /**
     * Copy ctor.
     *
     * @param src The object to clone from
     */
    JobInstanceRequest(const JobInstanceRequest& src);

    /**
     * Dtor.
     */
    virtual ~JobInstanceRequest(void);

    /**
     * Answer the description of the job to be instantiated.
     *
     * @return The description of the job to be instantiated
     */
    inline const JobDescription* Description(void) const {
        return this->desc;
    }

    /**
     * Sets the description of the job to be instantiated.
     *
     * @param desc The description of the job to be instantiated.
     */
    inline void SetDescription(const JobDescription* desc) {
        this->desc = desc;
    }

    /**
     * Assignment operator.
     *
     * @param rhs The right hand side operand.
     *
     * @return Reference to 'this'
     */
    JobInstanceRequest& operator=(const JobInstanceRequest& rhs);

    /**
     * Test for equality
     *
     * @param rhs The right hand side operand.
     *
     * @return 'true' if 'this' is equal to 'rhs'
     */
    bool operator==(const JobInstanceRequest& rhs) const;

private:
    /** The job description to be instantiated */
    const JobDescription* desc;
};

} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_JOBINSTANCEREQUEST_H_INCLUDED */
