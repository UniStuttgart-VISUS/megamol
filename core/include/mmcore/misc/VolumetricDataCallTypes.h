/*
 * VolumetricDataCallTypes.h
 *
 * Copyright (C) 2014 by Visualisierungsinstitut der Universität Stuttgart.
 * Alle rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_VOLUMETRICDATACALLTYPES_H_INCLUDED
#define MEGAMOLCORE_VOLUMETRICDATACALLTYPES_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include <cstring>


namespace megamol {
namespace core {
namespace misc {

    /** Possible type of grids. */
    enum GridType_t {
        NONE,
        CARTESIAN,
        RECTILINEAR,
        TETRAHEDRAL
    };

    /** Possible types of scalars. */
    enum ScalarType_t {
        UNKNOWN,
        SIGNED_INTEGER,
        UNSIGNED_INTEGER,
        FLOATING_POINT,
        BITS
    };

    /** Structure containing all required metadata about a data set. */
    struct VolumetricMetadata_t {

        /** Initialise a new instance. */
        inline VolumetricMetadata_t(void) {
            ::memset(this->Resolution, 0, sizeof(this->Resolution));
            ::memset(this->SliceDists, 0, sizeof(this->SliceDists));
			::memset(this->Origin, 0, sizeof(this->Origin));
            ::memset(this->IsUniform, 0, sizeof(this->IsUniform));
            ::memset(this->Extents, 0, sizeof(this->Extents));
        };

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
        float *SliceDists[3];

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
    };

} /* end namespace misc */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_VOLUMETRICDATACALLTYPES_H_INCLUDED */
