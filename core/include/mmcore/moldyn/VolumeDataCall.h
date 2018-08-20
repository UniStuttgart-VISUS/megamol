/*
 * VolumeDataCall.h
 *
 * Copyright (C) 2012 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */


#ifndef MEGAMOL_CORE_VOLUMEDATACALL_H_INCLUDED
#define MEGAMOL_CORE_VOLUMEDATACALL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/AbstractGetData3DCall.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "vislib/IllegalParamException.h"
#include "vislib/math/Cuboid.h"

#ifdef _WIN32
#define DEPRECATED __declspec(deprecated)
#else /* _WIN32 */
#define DEPRECATED
#endif /* _WIN32 */

namespace megamol {
namespace core {
namespace moldyn {

    /**
     * Base class of rendering graph calls and data interfaces for volumetric data.
     */
    class MEGAMOLCORE_API VolumeDataCall : public megamol::core::AbstractGetData3DCall {
    public:

        /** Index of the 'GetData' function */
        static const unsigned int CallForGetData;

        /** Index of the 'GetExtent' function */
        static const unsigned int CallForGetExtent;
        
        /**
         * Answer the name of the objects of this description.
         *
         * @return The name of the objects of this description.
         */
        static const char *ClassName(void) {
            return "VolumeDataCall";
        }

        /**
         * Gets a human readable description of the module.
         *
         * @return A human readable description of the module.
         */
        static const char *Description(void) {
            return "Call to get 3D volume data";
        }

        /**
         * Answer the number of functions used for this call.
         *
         * @return The number of functions used for this call.
         */
        static unsigned int FunctionCount(void) {
            return 2;
        }

        /**
         * Answer the name of the function used for this call.
         *
         * @param idx The index of the function to return it's name.
         *
         * @return The name of the requested function.
         */
        static const char * FunctionName(unsigned int idx) {
            switch(idx) {
                case 0:
                    return "GetData";
                case 1:
                    return "GetExtent";
            }
            return "";
        }

        /**
         * Sets the bounding box.
         *
         * @param minX minimal X value.
         * @param minY minimal Y value.
         * @param minZ minimal Z value.
         * @param maxX maximal X value.
         * @param maxY maximal Y value.
         * @param maxZ maximal Z value.
         */
        inline void SetBoundingBox(float minX, float minY, float minZ, float maxX, float maxY, float maxZ) {
            this->bBox.Set(minX, minY, minZ, maxX, maxY, maxZ);
            if (this->bBox.IsEmpty()) {
                this->bBox.Set(0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f);
            }
        }

        /**
         * Sets the bounding box.
         *
         * @param box The bounding box cuboid.
         */
        inline void SetBoundingBox(const vislib::math::Cuboid<float> &box) {
            this->bBox = box;
            if (this->bBox.IsEmpty()) {
                this->bBox.Set(0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f);
            }
        }

        /**
         * Answer the bounding box.
         *
         * @return the bounding box.
         */
        inline const vislib::math::Cuboid<float>& BoundingBox(void) const {
            return this->bBox;
        }

        /**
         * Sets the volume dimension (i.e. the resolution of the volume).
         *
         * @param cols Number of columns.
         * @param rows Number of rows.
         * @param secs Number of sections.
         */
        inline void SetVolumeDimension(unsigned int cols, unsigned int rows, unsigned int secs) {
            this->volDim.Set(cols, rows, secs);
        }

        /**
         * Sets the volume dimension (i.e. the resolution of the volume).
         *
         * @param box The bounding box cuboid.
         */
        inline void SetVolumeDimension(const vislib::math::Dimension<unsigned int, 3> &dim) {
            this->volDim = dim;
        }

        /**
         * Answer the volume dimension (i.e. the resolution of the volume).
         *
         * @return the bounding box.
         */
        inline const vislib::math::Dimension<unsigned int, 3>& VolumeDimension(void) const {
            return this->volDim;
        }

        /**
         * Set the minimum density value.
         *
         * @param density The minimum density.
         */
        inline void SetMinimumDensity(float density) {
            this->minDensity = density;
        }

        /**
         * Answer the minimum density value.
         *
         * @return The minimum density.
         */
        inline float MinimumDensity(void) const {
            return this->minDensity;
        }

        /**
         * Set the maximum density value.
         *
         * @param density The maximum density.
         */
        inline void SetMaximumDensity(float density) {
            this->maxDensity = density;
        }

        /**
         * Answer the maximum density value.
         *
         * @return The maximum density.
         */
        inline float MaximumDensity(void) const {
            return this->maxDensity;
        }

        /**
         * Set the mean (or average) density value.
         *
         * @param density The mean density.
         */
        inline void SetMeanDensity(float density) {
            this->meanDensity = density;
        }

        /**
         * Answer the mean (or average) density value.
         *
         * @return The mean density.
         */
        inline float MeanDensity(void) const {
            return this->meanDensity;
        }

        /**
         * Set the components per voxel.
         *
         * @param The components per voxel.
         */
        inline void SetComponents(unsigned int components) {
            this->components = components;
        }

        /**
         * Answer the components per voxel.
         *
         * @return The components per voxel.
         */
        inline unsigned int Components(void) const {
            return this->components;
        }

        /**
         * Sets a pointer to the voxel map array.
         * The memory pointed to will remain owned by the caller and the caller
         * must ensure it remains valid as long as it is used through this
         * interface.
         *
         * @param voxelMap Pointer to the voxel map array.
         */
        void SetVoxelMapPointer(float *voxelMap);

        /**
         * Returns a pointer to the voxel map. 
         * The pointer points to the internal memory structure. The caller 
         * must not free or alter the memory returned.
         *
         * @return A pointer to the voxel map.
         */
        inline const float* VoxelMap(void) const {
            return this->map;
        }

        /** Ctor. */
        VolumeDataCall(void);

        /** Dtor. */
        virtual ~VolumeDataCall(void);
          
    private:
        /** map of the volume */
        float *map;
        /** Flag whether or not the interface owns the memory of the voxel map */
        bool mapMemory;

#ifdef _WIN32
#pragma warning (disable: 4251)
#endif /* _WIN32 */
        /** bounding box */
        vislib::math::Cuboid<float> bBox;
#ifdef _WIN32
#pragma warning (default: 4251)
#endif /* _WIN32 */

        /** components per voxel */
        unsigned int components;

#ifdef _WIN32
#pragma warning (disable: 4251)
#endif /* _WIN32 */
        /** volume dimension (the resolution of the volume in voxels). */
        vislib::math::Dimension<unsigned int, 3> volDim;
#ifdef _WIN32
#pragma warning (default: 4251)
#endif /* _WIN32 */

        /** minimum voxel value */
        float minDensity;
        /** maximum voxel value */
        float maxDensity;
        /** mean voxel value */
        float meanDensity;

    };

    /** Description class typedef */
    typedef megamol::core::factories::CallAutoDescription<VolumeDataCall> VolumeDataCallDescription;


} /* end namespace moldyn */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOL_CORE_VOLUMEDATACALL_H_INCLUDED */
