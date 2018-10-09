/*
 * ASCIISphereLoader.h
 *
 * Copyright (C) 2016 by Karsten Schatz
 * Copyright (C) 2016 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MM101PLG_ASCIISPHERELOADER_H_INCLUDED
#define MM101PLG_ASCIISPHERELOADER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include <vector>
#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/String.h"
#include "vislib/math/Cuboid.h"

namespace megamol {
namespace megamol101 {

/**
 * Loader module for simple ASCII comma seperated files
 * which contain data for single spheres for each line.
 */
class ASCIISphereLoader : public core::Module {
public:
    /**
     * Answer the name of the objects of this description.
     *
     * TUTORIAL: Mandatory method for every module or call that states the name of the class.
     * This name should be unique.
     *
     * @return The name of the objects of this description.
     */
    static const char* ClassName(void) { return "ASCIISphereLoader"; }

    /**
     * Gets a human readable description of the module.
     *
     * TUTORIAL: Mandatory method for every module or call that returns a description.
     *
     * @return A human readable description of the module.
     */
    static const char* Description(void) {
        return "Loads sphere data from a csv file that contains one sphere per line.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * TUTORIAL: Mandatory method for every module.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    /** Constructor */
    ASCIISphereLoader(void);

    /** Destructor */
    virtual ~ASCIISphereLoader(void);

private:
    /**
     * Ensures that the data is loaded
     */
    void assertData(void);

    /**
     * Implementation of 'Create'.
     *
     * TUTORIAL: The overwritten version of this method gets called right after an object of this class has been
     *instantiated.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    /**
     * Gets the data from the source.
     *
     * TUTORIAL: This method computes the final data and writes them to the calling call.
     * Data that was written by getExtentCallback should be written again, just in case...
     *
     * @param caller The calling call.
     * @return 'true' on success, 'false' on failure.
     */
    virtual bool getDataCallback(core::Call& caller);

    /**
     * Gets the data extents from the source.
     *
     * TUTORIAL: This method computes the extents of the data set, namely the bounding box and other relevant values,
     * and writes them into the calling call.
     *
     * @param caller The calling call.
     * @return 'true' on success, 'false' on failure.
     */
    virtual bool getExtentCallback(core::Call& caller);

    /**
     * Loads the specified file
     *
     * @param filename The file to load
     * @return 'true' on success, 'false' on failure.
     */
    bool load(const vislib::TString& filename);

    /**
     * Implementation of 'Release'.
     */
    virtual void release(void);

    /** The file name. */
    core::param::ParamSlot filenameSlot;

    /** The slot for requesting data */
    core::CalleeSlot getDataSlot;

    /** The data update hash */
    SIZE_T datahash;

    /** The bounding box */
    vislib::math::Cuboid<float> bbox;

    /** Pointer to the sphere parameters. */
    std::vector<float> spheres;

    /** The total number of loaded spheres */
    SIZE_T numSpheres;
};

} /* end namespace megamol101 */
} /* end namespace megamol */

#endif /* MM101PLG_ASCIISPHERELOADER_H_INCLUDED */
