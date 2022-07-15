
/*
 * CallVolumetricData.h
 *
 * Copyright (C) 2010 by Sebastian Grottel
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/factories/CallAutoDescription.h"
#include "mmstd/data/AbstractGetData3DCall.h"
#include "vislib/Array.h"
#include "vislib/String.h"
#include "vislib/assert.h"


namespace megamol {
namespace trisoup {

/**
 * Call transporting data from volumetrics ...
 */
class trisoupVolumetricDataCall : public core::AbstractGetData3DCall {
public:
    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char* ClassName(void) {
        return "trisoupVolumetricDataCall";
    }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description(void) {
        return "Call transporting volumetric data";
    }

    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount(void) {
        return AbstractGetData3DCall::FunctionCount();
    }

    /**
     * Answer the name of the function used for this call.
     *
     * @param idx The index of the function to return it's name.
     *
     * @return The name of the requested function.
     */
    static const char* FunctionName(unsigned int idx) {
        return AbstractGetData3DCall::FunctionName(idx);
    }

    /** Ctor */
    trisoupVolumetricDataCall(void) {}

    /** Dtor */
    virtual ~trisoupVolumetricDataCall(void) {}

public:
    typedef char VoxelType;
    class Volume {
    public:
        Volume() : volumeData(0) {}

        VoxelType* volumeData;
        int resX, resY, resZ;
        double origin[3], scaling[3];
        // dirty hack
        inline int operator==(const Volume& v) {
            return v.volumeData == this->volumeData;
        }
        VISLIB_FORCEINLINE unsigned cellIndex(unsigned x, unsigned y, unsigned z) {
            return (z * this->resY + y) * this->resX + x;
        }
        /*    VISLIB_FORCEINLINE bool isBorder(unsigned x, unsigned y, unsigned z) {
                return (x == 0) || (x == this->resX - 2)
                    || (y == 0) || (y == this->resY - 2)
                    || (z == 0) || (z == this->resZ - 2);
            }*/
    };

    void SetVolumes(vislib::Array<Volume>& volumes) {
        this->subVolumes = volumes;
    }
    vislib::Array<Volume>& GetVolumes() {
        return subVolumes;
    }

private:
    vislib::Array<Volume> subVolumes;
};

} // namespace trisoup
} // namespace megamol
