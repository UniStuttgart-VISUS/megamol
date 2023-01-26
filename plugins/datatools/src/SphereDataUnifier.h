/*
 * SphereDataUnifier.h
 *
 * Copyright (C) 2012 by TU Dresden
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_SPHEREDATAUNIFIER_H_INCLUDED
#define MEGAMOLCORE_SPHEREDATAUNIFIER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "vislib/RawStorage.h"
#include "vislib/math/Cuboid.h"


namespace megamol {
namespace datatools {


/**
 * Renderer for gridded imposters
 */
class SphereDataUnifier : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "SphereDataUnifier";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Unifies Sphere Data. (Evil-Implementation Warning)";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    /** Ctor. */
    SphereDataUnifier();

    /** Dtor. */
    ~SphereDataUnifier() override;

private:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

    /**
     * Implementation of 'Release'.
     */
    void release() override;

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

    /** TODO: Document */
    void accumExt(bool& first, float x, float y, float z, float r);

    /** The call for the output data */
    core::CalleeSlot putDataSlot;

    /** The call for the input data */
    core::CallerSlot getDataSlot;

    /** The in data hash */
    SIZE_T inDataHash;

    /** The out data hash */
    SIZE_T outDataHash;

    /** The generated data */
    vislib::RawStorage data;

    /** The new bounding box */
    vislib::math::Cuboid<float> bbox;

    /** The new clip box */
    vislib::math::Cuboid<float> cbox;
};


} /* end namespace datatools */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_SPHEREDATAUNIFIER_H_INCLUDED */
