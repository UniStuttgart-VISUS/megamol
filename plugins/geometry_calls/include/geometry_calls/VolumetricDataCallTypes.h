/*
 * VolumetricDataCallTypes.h
 *
 * Copyright (C) 2014-2018 by MegaMol Team.
 * Alle rechte vorbehalten.
 */

#pragma once

#include <cstring>


namespace megamol::geocalls {

/** Possible type of grids. */
enum GridType_t { NONE, CARTESIAN, RECTILINEAR, TETRAHEDRAL };

/** Possible types of scalars. */
enum ScalarType_t { UNKNOWN, SIGNED_INTEGER, UNSIGNED_INTEGER, FLOATING_POINT, BITS };

/** Possible (physical) memory locations */
enum MemoryLocation { VRAM, RAM };

/**
 * Structure containing all required metadata about a data set, which are
 * natively stored by the datRaw library (the structure allows for zero-copy
 * transfer of the metadata).
 *
 * Use a VolumetricMetadataStore if you need to compute the metadata
 * yourself instead of obtaining them from the datraw library.
 */
struct VolumetricMetadata_t {

    /** Initialise a new instance. */
    VolumetricMetadata_t(void) {
        ::memset(this->Resolution, 0, sizeof(this->Resolution));
        ::memset(this->SliceDists, 0, sizeof(this->SliceDists));
        ::memset(this->Origin, 0, sizeof(this->Origin));
        ::memset(this->IsUniform, 0, sizeof(this->IsUniform));
        ::memset(this->Extents, 0, sizeof(this->Extents));
        MinValues = nullptr;
        MaxValues = nullptr;
        MemLoc = RAM;
    }

    // creates a deep copy of the instance. beware that the owner of the copy
    // needs to deallocate the components of SliceDists and all of MinValues and MaxValues explicitly!
    VolumetricMetadata_t Clone() const {
        VolumetricMetadata_t clone;
        clone.Components = this->Components;
        clone.GridType = this->GridType;
        clone.ScalarType = this->ScalarType;
        clone.ScalarLength = this->ScalarLength;
        clone.NumberOfFrames = this->NumberOfFrames;
        memcpy(clone.Resolution, this->Resolution, sizeof(size_t) * 3);
        memcpy(clone.Origin, this->Origin, sizeof(float) * 3);
        memcpy(clone.IsUniform, this->IsUniform, sizeof(bool) * 3);
        memcpy(clone.Extents, this->Extents, sizeof(float) * 3);

        if (this->GridType == RECTILINEAR) {
            for (auto x = 0; x < 3; ++x) {
                clone.SliceDists[x] = new float[this->Resolution[x] - 1];
                memcpy(clone.SliceDists[x], this->SliceDists[x], sizeof(float) * (this->Resolution[x] - 1));
            }
        } else {
            for (auto x = 0; x < 3; ++x) {
                clone.SliceDists[x] = new float[1];
                *clone.SliceDists[x] = *this->SliceDists[x];
            }
        }
        clone.MinValues = new double[this->Components];
        clone.MaxValues = new double[this->Components];
        memcpy(clone.MinValues, this->MinValues, sizeof(double) * this->Components);
        memcpy(clone.MaxValues, this->MaxValues, sizeof(double) * this->Components);
        clone.MemLoc = this->MemLoc;
        return clone;
    }

    /** The type of the grid. */
    enum GridType_t GridType;

    /** The resolution of the three dimensions. */
    size_t Resolution[3];

    /** The type of a scalar. */
    enum ScalarType_t ScalarType;

    /** The length of a scalar in bytes. */
    size_t ScalarLength;

    /** The number of components per grid point. */
    size_t Components;

    /**
     * The distance between slices each of the three dimensions. The
     * data source providing the metadata remains owner of the arrays.
     */
    float* SliceDists[3];

    /**
     * The origin of the coordinate system used by this data set
     */
    float Origin[3];

    /**
     * Determines whether SliceDists[i] is uniform and has only one
     * entry.
     */
    bool IsUniform[3];

    /** The total number of frames in the data set. */
    size_t NumberOfFrames;

    /**
     * The extents of the data set, taking into account that the slices
     * might have different distances.
     */
    float Extents[3];

    /**
     * Minimal values per component.
     */
    double* MinValues;

    /**
     * Maximal values per component.
     */
    double* MaxValues;

    /**
     * (Physical) memory location of the volume data.
     */
    enum MemoryLocation MemLoc;
};

} // namespace megamol::geocalls
