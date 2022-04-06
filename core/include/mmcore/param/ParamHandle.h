/*
 * ParamHandle.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_PARAMHANDLE_H_INCLUDED
#define MEGAMOLCORE_PARAMHANDLE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "AbstractParam.h"
#include "mmcore/ApiHandle.h"
#include "mmcore/CoreInstance.h"
#include "vislib/SmartPtr.h"


namespace megamol {
namespace core {
namespace param {


/**
 * Wrapper class for parameter handles.
 */
class ParamHandle : public ApiHandle {
public:
    /**
     * Ctor.
     * Creates a new handle for the given parameter smart pointer.
     *
     * @param inst The owning core instance.
     * @param param The smart pointer to the parameter object.
     */
    ParamHandle(const CoreInstance& inst, const vislib::SmartPtr<AbstractParam>& param);

    /**
     * Copy ctor.
     *
     * @param src The object to clone from.
     */
    ParamHandle(const ParamHandle& src);

    /**
     * Dtor.
     */
    virtual ~ParamHandle(void);

    /**
     * Gets the owning core instance.
     *
     * @return The owning core instance.
     */
    inline const CoreInstance& GetCoreInstance(void) const {
        return this->inst;
    }

    /**
     * Gets the id string of this job.
     *
     * @param outID receives the id string of this job.
     */
    void GetIDString(vislib::StringA& outID);

    /**
     * Gets the id string of this job.
     *
     * @param outID receives the id string of this job.
     */
    inline void GetIDString(vislib::StringW& outID) {
        vislib::StringA tmp;
        this->GetIDString(tmp);
        outID = tmp;
    }

    /**
     * Gets the encapsuled parameter object.
     *
     * @return The encapsuled parameter object.
     */
    inline vislib::SmartPtr<AbstractParam> GetParameter(void) const {
        return this->param;
    }

private:
    /** Pointer to the owning core instance */
    const CoreInstance& inst;

#ifdef _WIN32
#pragma warning(disable : 4251)
#endif /* _WIN32 */
    /** smart pointer to the real parameter object */
    vislib::SmartPtr<AbstractParam> param;
#ifdef _WIN32
#pragma warning(default : 4251)
#endif /* _WIN32 */
};


} /* end namespace param */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_PARAMHANDLE_H_INCLUDED */
