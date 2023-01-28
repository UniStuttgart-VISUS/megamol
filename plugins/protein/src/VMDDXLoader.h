//
// VMDDXLoader.h
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: May 7, 2013
//     Author: scharnkn
//

//
// VTILoader.h
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: Apr 12, 2013
//     Author: scharnkn
//

#ifndef MMPROTEINPLUGIN_VMDDXLOADER_H_INCLUDED
#define MMPROTEINPLUGIN_VMDDXLOADER_H_INCLUDED
#pragma once

#include "HostArr.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "protein_calls/VTIDataCall.h"
#include "protein_calls/VTKImageData.h"
#include "vislib/Map.h"
#include "vislib/String.h"
#include "vislib/math/Cuboid.h"
#include "vislib/math/Vector.h"

#include <fstream>
#include <map>

typedef unsigned int uint;

namespace megamol::protein {

class VMDDXLoader : public megamol::core::Module {
public:
    /** Ctor */
    VMDDXLoader();

    /** Dtor */
    ~VMDDXLoader() override;

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "VMDDXLoader";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Loader module for *.dx file format used by the VMD.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

    /**
     * Implementation of 'release'.
     */
    void release() override;

    /**
     * Call callback to get the data
     *
     * @param call The calling call
     * @return True on success
     */
    bool getData(core::Call& call);

    /**
     * Call callback to get the extent of the data
     *
     * @param call The calling call
     * @return True on success
     */
    bool getExtent(core::Call& call);

    /**
     * Loads a *.vti file.
     *
     * @param filename The files name.
     * @return 'True' on success, 'false', otherwise
     */
    bool loadFile(const vislib::StringA& filename);

    /**
     * Reads data from an ASCII buffer to a float array.
     *
     * @param buffIn  The input buffer containing the ascii data.
     * @param buffOut The output buffer containing the floats
     * @param sizeOut The size of the output buffer
     */
    void readDataAscii2Float(char* buffIn, float* buffOut, SIZE_T sizeOut);

private:
    /** TODO */
    void scanFolder();

    /**
     * Convert a string to the respective integer value using string stream
     * objects.
     *
     * @param str The string
     * @return The integer
     */
    int string2int(vislib::StringA str);

    /**
     * Convert a string to the respective float value using string stream
     * objects.
     *
     * @param str The string
     * @return The float
     */
    float string2float(vislib::StringA str);


    /// The data callee slot
    core::CalleeSlot dataOutSlot;


    /* Parameters slots */

    /// Parameter slot containing path to the data file
    core::param::ParamSlot filenameSlot;


    /* Data set */

    SIZE_T hash; ///> The data hash

    //    Cubei extent;    ///> The data sets number of elements in each direction
    //    Vec3f origin;    ///> The data sets origin in world space coordinates
    //    Vec3f spacing;   ///> The data sets spacing in each direction
    protein_calls::VTKImageData imgdata; ///> The image data object

    vislib::TString filenamesPrefix; ///> Prefix for the data files
    vislib::TString filenamesSuffix; ///> Suffix for the data files
    uint filenamesDigits;            ///> Number of digits in the data set

    int nFrames; ///> The number of frames in the data set

    HostArr<float> data;    ///> Pointer to the data
    HostArr<float> dataTmp; ///> Pointer to the temporary data
    //    float min;               ///> Minimum value of the data
    //    float max;               ///> Minimum value of the data
};


} // namespace megamol::protein

#endif // MMPROTEINPLUGIN_VMDDXLOADER_H_INCLUDED
