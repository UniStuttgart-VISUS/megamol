/*
 * CallBinaryVolumeData.h
 *
 * Copyright (C) 2010 by Sebastian Grottel
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MMTRISOUPPLG_CALLBINARYVOLUMEDATAA_H_INCLUDED
#define MMTRISOUPPLG_CALLBINARYVOLUMEDATAA_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/factories/CallAutoDescription.h"
#include "mmstd/data/AbstractGetData3DCall.h"
#include "vislib/Array.h"
#include "vislib/String.h"
#include "vislib/assert.h"


namespace megamol {
namespace trisoup {

/**
 * Call transporting binary volume data
 *
 * The volume is stored as linear array of bool values:
 *     volume[x + y * xSize + z * xSize * ySize]
 */
class CallBinaryVolumeData : public core::AbstractGetData3DCall {
public:
    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char* ClassName(void) {
        return "CallBinaryVolumeData";
    }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description(void) {
        return "Call transporting binary volume data";
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
    CallBinaryVolumeData(void);

    /** Dtor */
    virtual ~CallBinaryVolumeData(void);

    /**
     * Gets the value of a single voxel
     *
     * @param x The x coordinate
     * @param y The y coordinate
     * @param z The z coordinate
     *
     * @return The voxel value or false if the coordinates are outside the volume
     */
    bool GetSafeVoxel(unsigned int x, unsigned int y, unsigned int z) const;

    /**
     * Gets the value of a single voxel
     *
     * @param x The x coordinate
     * @param y The y coordinate
     * @param z The z coordinate
     *
     * @return The voxel value or false if the coordinates are outside the volume
     */
    inline bool GetSafeVoxel(int x, int y, int z) const {
        return ((x < 0) || (y < 0) || (z < 0)) ? false
                                               : this->GetSafeVoxel(static_cast<unsigned int>(x),
                                                     static_cast<unsigned int>(y), static_cast<unsigned int>(z));
    }

    /**
     * Answer the number of voxels in x direction
     *
     * @return The number of voxels in x direction
     */
    inline unsigned int GetSizeX(void) const {
        return this->sizeX;
    }

    /**
     * Answer the number of voxels in y direction
     *
     * @return The number of voxels in y direction
     */
    inline unsigned int GetSizeY(void) const {
        return this->sizeY;
    }

    /**
     * Answer the number of voxels in z direction
     *
     * @return The number of voxels in z direction
     */
    inline unsigned int GetSizeZ(void) const {
        return this->sizeZ;
    }

    /**
     * Answer the size of a single voxel in x direction
     *
     * @return The size of a single voxel in x direction
     */
    inline float GetVoxelSizeX(void) const {
        return this->voxSizeX;
    }

    /**
     * Answer the size of a single voxel in y direction
     *
     * @return The size of a single voxel in y direction
     */
    inline float GetVoxelSizeY(void) const {
        return this->voxSizeY;
    }

    /**
     * Answer the size of a single voxel in z direction
     *
     * @return The size of a single voxel in z direction
     */
    inline float GetVoxelSizeZ(void) const {
        return this->voxSizeZ;
    }

    /**
     * Answer the volume pointer
     *
     * @return The volume pointer
     */
    inline const bool* GetVolume(void) const {
        return this->volume;
    }

    /**
     * Sets the volume data
     *
     * @param sizeX Number of voxels in x direction
     * @param sizeY Number of voxels in y direction
     * @param sizeZ Number of voxels in z direction
     * @param volume Pointer to the bool volume
     */
    inline void SetVolume(unsigned int sizeX, unsigned int sizeY, unsigned int sizeZ, bool* volume) {
        this->sizeX = sizeX;
        this->sizeY = sizeY;
        this->sizeZ = sizeZ;
        this->voxSizeX = 1.0f;
        this->voxSizeY = 1.0f;
        this->voxSizeZ = 1.0f;
        this->volume = volume;
    }

    /**
     * Sets the volume data
     *
     * @param sizeX Number of voxels in x direction
     * @param sizeY Number of voxels in y direction
     * @param sizeZ Number of voxels in z direction
     * @param voxSizeX The size of a single voxel in x direction
     * @param voxSizeY The size of a single voxel in y direction
     * @param voxSizeZ The size of a single voxel in z direction
     * @param volume Pointer to the bool volume
     */
    inline void SetVolume(unsigned int sizeX, unsigned int sizeY, unsigned int sizeZ, float voxSizeX, float voxSizeY,
        float voxSizeZ, bool* volume) {
        this->sizeX = sizeX;
        this->sizeY = sizeY;
        this->sizeZ = sizeZ;
        this->voxSizeX = voxSizeX;
        this->voxSizeY = voxSizeY;
        this->voxSizeZ = voxSizeZ;
        this->volume = volume;
    }

private:
    /** The size of the volume in voxel counts */
    unsigned int sizeX, sizeY, sizeZ;

    /** The size of each side of a single voxel */
    float voxSizeX, voxSizeY, voxSizeZ;

    /** The volume data */
    bool* volume;
};

/** Description class typedef */
typedef core::factories::CallAutoDescription<CallBinaryVolumeData> CallBinaryVolumeDataDescription;

} /* end namespace trisoup */
} /* end namespace megamol */

#endif /* MMTRISOUPPLG_CALLBINARYVOLUMEDATAA_H_INCLUDED */
