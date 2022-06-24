/*
 * AbstractSimpleParticleDataSource.h
 *
 * Copyright (C) 2012 by TU Dresden (CGV)
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "geometry_calls/MultiParticleDataCall.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"


namespace megamol::datatools::io {


/**
 * Abstract base class for simple particle loaders (single time step = no animation)
 */
class AbstractSimpleParticleDataSource : public core::Module {
public:
protected:
    /** Ctor. */
    AbstractSimpleParticleDataSource(void);

    /** Dtor. */
    virtual ~AbstractSimpleParticleDataSource(void);

    /**
     * Gets the data from the source.
     *
     * @param call The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    virtual bool getData(geocalls::MultiParticleDataCall& call) = 0;

    /**
     * Gets the data from the source.
     *
     * @param call The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    virtual bool getExtent(geocalls::MultiParticleDataCall& call) = 0;

    /** The file name */
    core::param::ParamSlot filenameSlot;

private:
    /**
     * Gets the data from the source.
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool getDataCallback(core::Call& caller);

    /**
     * Gets the data from the source.
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool getExtentCallback(core::Call& caller);

    /** The slot for requesting data */
    core::CalleeSlot getDataSlot;
};


} // namespace megamol::datatools::io
