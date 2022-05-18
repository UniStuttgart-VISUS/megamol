/*
 * InstanceRequest.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_INSTANCEREQUEST_H_INCLUDED
#define MEGAMOLCORE_INSTANCEREQUEST_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/ParamValueSetRequest.h"
#include "vislib/String.h"


namespace megamol {
namespace core {

/**
 * Abstract base class of job and view instantiation requests.
 */
class InstanceRequest : public ParamValueSetRequest {
public:
    /**
     * Answer the name for the instance to be instantiated.
     *
     * @return The name
     */
    inline const vislib::StringA& Name(void) const {
        return this->name;
    }

    /**
     * Sets the name for the instance to be instantiated.
     *
     * @param name The new name
     */
    inline void SetName(const vislib::StringA& name) {
        this->name = name;
    }

protected:
    /**
     * Ctor.
     */
    InstanceRequest(void);

    /**
     * Dtor.
     */
    virtual ~InstanceRequest(void);

private:
    /** The name of the instance requested. */
    vislib::StringA name;
};

} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_INSTANCEREQUEST_H_INCLUDED */
