/*
 * JobDescription.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_JOBDESCRIPTION_H_INCLUDED
#define MEGAMOLCORE_JOBDESCRIPTION_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/InstanceDescription.h"


namespace megamol {
namespace core {

/**
 * Class of job descriptions.
 */
class JobDescription : public InstanceDescription {
public:
    /**
     * Ctor.
     *
     * @param classname The name of the job described.
     */
    JobDescription(const char* classname);

    /**
     * Dtor.
     */
    virtual ~JobDescription(void);

    /**
     * Sets the id of the module to be used as job controller module of
     * this job.
     *
     * @param id The id of the job control module.
     */
    inline void SetJobModuleID(const vislib::StringA& id) {
        this->jobModID = id;
    }

    /**
     * Gets the id of the module to be used as job control module of this
     * job.
     *
     * @return The id of the job control module.
     */
    inline const vislib::StringA& JobModuleID(void) const {
        return this->jobModID;
    }

private:
#ifdef _WIN32
#pragma warning(disable : 4251)
#endif /* _WIN32 */
    /** The id of the module to be used as view */
    vislib::StringA jobModID;
#ifdef _WIN32
#pragma warning(default : 4251)
#endif /* _WIN32 */
};

} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_JOBDESCRIPTION_H_INCLUDED */
