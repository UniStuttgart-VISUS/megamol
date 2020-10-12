/*
 * MSMSGenus0Generator.h
 *
 * Copyright (C) 2015 by Michael Krone
 * Copyright (C) 2015 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MMMOLMAPPLG_MSMSGENUS0GENERATOR_H_INCLUDED
#define MMMOLMAPPLG_MSMSGENUS0GENERATOR_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "geometry_calls/CallTriMeshData.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/Array.h"
#include "vislib/math/Cuboid.h"


namespace megamol {
namespace protein {
/**
 * Class for loading MSMS mesh data
 */
class MSMSGenus0Generator : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "MSMSGenus0Generator"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Data source for MSMS mesh data files that always generates a genus 0 mesh";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    /** Ctor */
    MSMSGenus0Generator(void);

    /** Dtor */
    virtual ~MSMSGenus0Generator(void);

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    /**
     * Implementation of 'Release'.
     */
    virtual void release(void);

    /** The bounding box */
    vislib::math::Cuboid<float> bbox;

    /** The data update hash */
    SIZE_T datahash;

    /**
     * Loads the specified file
     *
     * @param filename The file to load
     * @param probe_radius The probe radius
     * @param frameID  The frame ID
     *
     * @return True on success
     */
    virtual bool load(const vislib::TString& filename, float probe_radius, unsigned int frameID = 0);

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

    /**
     * Computes the genus of the stored mesh and returns true if it is 0
     *
     * @param frame The id of the requested frame
     * @return True if the genus is 0, false otherwise
     */
    bool isGenus0(uint32_t frameID, uint32_t* outGenus = nullptr);

    /** The slot for requesting data */
    core::CalleeSlot getDataSlot;

    /** The slot for getting protein data */
    core::CallerSlot molDataSlot;

    /** The slot for getting binding site data */
    core::CallerSlot bsDataSlot;

    /** The slot for getting per-atom float data */
    core::CallerSlot perAtomDataSlot;

    /** the file name slot */
    core::param::ParamSlot filenameSlot;
    /** parameter slot for color table filename */
    megamol::core::param::ParamSlot colorTableFileParam;
    /** parameter slot for coloring mode */
    megamol::core::param::ParamSlot coloringModeParam0;
    /** parameter slot for coloring mode */
    megamol::core::param::ParamSlot coloringModeParam1;
    /** parameter slot for coloring mode weighting*/
    megamol::core::param::ParamSlot colorWeightParam;
    /** parameter slot for min color of gradient color mode */
    megamol::core::param::ParamSlot minGradColorParam;
    /** parameter slot for mid color of gradient color mode */
    megamol::core::param::ParamSlot midGradColorParam;
    /** parameter slot for max color of gradient color mode */
    megamol::core::param::ParamSlot maxGradColorParam;
    /** MSMS detail parameter */
    megamol::core::param::ParamSlot msmsDetailParam;
    /** MSMS detail parameter */
    megamol::core::param::ParamSlot msmsStartingRadiusParam;
    /** MSMS step size parameter */
    megamol::core::param::ParamSlot msmsStepSizeParam;
    /** MSMS numTrysParam */
    megamol::core::param::ParamSlot msmsMaxTryNumParam;

    /** The color lookup table (for chains, amino acids,...) */
    vislib::Array<vislib::math::Vector<float, 3>> colorLookupTable;
    /** The color lookup table which stores the rainbow colors */
    vislib::Array<vislib::math::Vector<float, 3>> rainbowColors;

    /** the number of vertices */
    unsigned int vertexCount;
    /** the number of faces */
    unsigned int faceCount;
    /** the number of atoms */
    unsigned int atomCount;

    /** the index of the vertex attribute */
    unsigned int attIdx;

    vislib::Array<geocalls::CallTriMeshData::Mesh*> obj;

    int prevTime;
};

} // namespace protein
} /* end namespace megamol */

#endif /* MMMOLMAPPLG_MSMSMESHLOADER_H_INCLUDED */
