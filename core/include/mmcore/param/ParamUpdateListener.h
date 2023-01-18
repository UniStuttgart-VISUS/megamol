/**
 * MegaMol
 * Copyright (c) 2010, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <string>
#include <vector>

#include "mmcore/param/ParamSlot.h"

namespace megamol::core::param {

/**
 * Abstract base class for all parameter objects
 */
class ParamUpdateListener {
public:
    using param_updates_vec_t = std::vector<std::pair<std::string, std::string>>;

    /** Ctor */
    ParamUpdateListener();

    /** Dtor. */
    virtual ~ParamUpdateListener();

    /**
     * Callback called when a parameter is updated
     *
     * @param slot The parameter updated
     */
    virtual void ParamUpdated(ParamSlot& slot) = 0;

    /**
     * Callback called to communicate a batch of parameter updates
     *
     * @param updates Vector containing all parameter updates to communicate
     */
    virtual void BatchParamUpdated(param_updates_vec_t const& updates);
};


} // namespace megamol::core::param
