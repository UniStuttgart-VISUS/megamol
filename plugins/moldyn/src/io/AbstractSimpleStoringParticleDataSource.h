/*
 * AbstractSimpleStoringParticleDataSource.h
 *
 * Copyright (C) 2012 by TU Dresden (CGV)
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "AbstractSimpleParticleDataSource.h"
#include "geometry_calls/MultiParticleDataCall.h"
#include "vislib/RawStorage.h"
#include "vislib/graphics/ColourRGBAu8.h"
#include "vislib/math/Cuboid.h"


namespace megamol::datatools::io {


/**
 * Abstract base class for simple particle loaders (single time step = no animation)
 */
class AbstractSimpleStoringParticleDataSource : public AbstractSimpleParticleDataSource {
public:
protected:
    /** Ctor. */
    AbstractSimpleStoringParticleDataSource(void);

    /** Dtor. */
    virtual ~AbstractSimpleStoringParticleDataSource(void);

    /**
     * Loads data if required to
     *
     * @param needLoad If true, then the data will be loaded anew
     */
    virtual void assertData(bool needLoad = false);

    /**
     * Loads the data
     *
     * @param filename The full path to the file to load
     *
     * @return True on success, false on failure
     */
    virtual bool loadData(const vislib::TString& filename) = 0;

    /**
     * Gets the data from the source.
     *
     * @param call The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    virtual bool getData(geocalls::MultiParticleDataCall& call);

    /**
     * Gets the data from the source.
     *
     * @param call The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    virtual bool getExtent(geocalls::MultiParticleDataCall& call);

#ifdef _WIN32
#pragma warning(disable : 4251)
#endif /* _WIN32 */

    /** The position data */
    vislib::RawStorage posData;

    /** The position data type */
    geocalls::MultiParticleDataCall::Particles::VertexDataType posDataType;

    /** The colour data */
    vislib::RawStorage colData;

    /** The colour data type */
    geocalls::MultiParticleDataCall::Particles::ColourDataType colDataType;

    /** The bounding box of positions */
    vislib::math::Cuboid<float> bbox;

    /** The clipping box of particles */
    vislib::math::Cuboid<float> cbox;

    /** The default colour to be used */
    vislib::graphics::ColourRGBAu8 defCol;

#ifdef _WIN32
#pragma warning(default : 4251)
#endif /* _WIN32 */

    /** The default radius */
    float defRad;

    /** The bounding values of the colour column */
    float minColVal, maxColVal;

    /** The hash value of the loaded data */
    SIZE_T datahash;

private:
};


} // namespace megamol::datatools::io
