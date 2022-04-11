/*
 * ViewDescription.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_VIEWDESCRIPTION_H_INCLUDED
#define MEGAMOLCORE_VIEWDESCRIPTION_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/InstanceDescription.h"


namespace megamol {
namespace core {

/**
 * Class of view descriptions.
 */
class ViewDescription : public InstanceDescription {
public:
    /**
     * Ctor.
     *
     * @param classname The name of the view described.
     */
    ViewDescription(const char* classname);

    /**
     * Dtor.
     */
    virtual ~ViewDescription(void);

    /**
     * Sets the id of the module to be used as view module of this view.
     *
     * @param id The id of the view module.
     */
    inline void SetViewModuleID(const vislib::StringA& id) {
        this->viewID = id;
    }

    /**
     * Gets the id of the module to be used as view module of this view.
     *
     * @return The id of the view module.
     */
    inline const vislib::StringA& ViewModuleID(void) const {
        return this->viewID;
    }

private:
    /** The id of the module to be used as view */
    vislib::StringA viewID;
};

} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_VIEWDESCRIPTION_H_INCLUDED */
