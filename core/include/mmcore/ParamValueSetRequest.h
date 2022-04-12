/*
 * ParamValueSetRequest.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_PARAMVALUESETREQUEST_H_INCLUDED
#define MEGAMOLCORE_PARAMVALUESETREQUEST_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "vislib/Array.h"
#include "vislib/Pair.h"
#include "vislib/String.h"


namespace megamol {
namespace core {

/**
 * Class managing parameter value set requests
 */
class ParamValueSetRequest {
public:
    /** Type for changing values for parameters */
    typedef vislib::Pair<vislib::StringA, vislib::TString> ParamValueRequest;

    /**
     * Ctor.
     */
    ParamValueSetRequest(void);

    /**
     * Copy ctor.
     *
     * @param src The object to clone from
     */
    ParamValueSetRequest(const ParamValueSetRequest& src);

    /**
     * Dtor.
     */
    virtual ~ParamValueSetRequest(void);

    /**
     * Add a parameter value pair
     *
     * @param name The relative name of the parameter to set
     * @param value The value to set the parameter to
     */
    inline void AddParamValue(const vislib::StringA& name, const vislib::TString& value) {
        this->paramValues.Add(ParamValueRequest(name, value));
    }

    /**
     * Clears the list of parameter value pairs.
     */
    inline void ClearParamValues(void) {
        this->paramValues.Clear();
    }

    /**
     * Gets the 'idx'-th parameter value request.
     *
     * @param idx The index of the parameter value to be returned
     *
     * @return The requested parameter value request
     */
    inline const ParamValueRequest& ParamValue(unsigned int idx) const {
        return this->paramValues[idx];
    }

    /**
     * Gets the number of stored parameter value requests.
     *
     * @return The number of parameter value requests.
     */
    inline unsigned int ParamValueCount(void) const {
        return static_cast<unsigned int>(this->paramValues.Count());
    }

    /**
     * Assignment operator.
     *
     * @param rhs The right hand side operand.
     *
     * @return Reference to 'this'
     */
    ParamValueSetRequest& operator=(const ParamValueSetRequest& rhs);

    /**
     * Test for equality
     *
     * @param rhs The right hand side operand.
     *
     * @return 'true' if 'this' is equal to 'rhs'
     */
    bool operator==(const ParamValueSetRequest& rhs) const;

private:
#ifdef _WIN32
#pragma warning(disable : 4251)
#endif /* _WIN32 */
    /** The instantiation parameter values */
    vislib::Array<ParamValueRequest> paramValues;
#ifdef _WIN32
#pragma warning(default : 4251)
#endif /* _WIN32 */
};

} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_PARAMVALUESETREQUEST_H_INCLUDED */
