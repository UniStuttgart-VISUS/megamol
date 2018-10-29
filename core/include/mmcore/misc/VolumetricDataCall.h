/*
 * VolumetricDataCall.h
 *
 * Copyright (C) 2014 by Visualisierungsinstitut der Universität Stuttgart.
 * Alle rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_VOLUMETRICDATACALL_H_INCLUDED
#define MEGAMOLCORE_VOLUMETRICDATACALL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/AbstractGetData3DCall.h"
#include "mmcore/factories/CallAutoDescription.h"

#include "mmcore/misc/VolumetricDataCallTypes.h"

#include "vislib/Array.h"
#include "vislib/sys/Log.h"


namespace megamol {
namespace core {
namespace misc {

    /**
     * Provides sampled volumetric data eg from the dat/raw library.
     */
    class MEGAMOLCORE_API VolumetricDataCall : public AbstractGetData3DCall {

    public:

        /** Possible type of grids. */
        typedef enum megamol::core::misc::GridType_t GridType;

        /** Possible types of scalars. */
        typedef enum megamol::core::misc::ScalarType_t ScalarType;

        /** Structure containing all required metadata about a data set. */
        typedef struct megamol::core::misc::VolumetricMetadata_t Metadata;

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static inline const char *ClassName(void) {
            return "VolumetricDataCall";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static inline const char *Description(void) {
            return "Transports volumetric data.";
        }

        /**
         * Answer the number of functions used for this call.
         *
         * @return The number of functions used for this call.
         */
        static unsigned int FunctionCount(void);

        /**
         * Answer the name of the function used for this call.
         *
         * @param idx The index of the function to return it's name.
         *
         * @return The name of the requested function.
         */
        static const char *FunctionName(unsigned int idx);

        /** Index of the function retrieving the data. */
        static const unsigned int IDX_GET_DATA;

        /** Index of the function retrieving the bounding box. */
        static const unsigned int IDX_GET_EXTENTS;

        /** Index of the function retrieving the meta data (dat content). */
        static const unsigned int IDX_GET_METADATA;

        /** Index of the function enabling asynchronous loading. */
        static const unsigned int IDX_START_ASYNC;

        /** Index of the function disabling asynchronous loading. */
        static const unsigned int IDX_STOP_ASYNC;

        /** Index of the function retrieving data that might be unavailable. */
        static const unsigned int IDX_TRY_GET_DATA;

        /**
         * Initialises a new instance.
         */
        VolumetricDataCall(void);

        /**
         * Clone 'rhs'.
         *
         * @param rhs The object to be cloned.
         */
        VolumetricDataCall(const VolumetricDataCall& rhs);

        /**
         * Finalises an instance.
         */
        virtual ~VolumetricDataCall(void);

        /**
         * Gets the number of frames starting at GetData().
         *
         * @return The number of frames loaded.
         */
        inline size_t GetAvailableFrames(void) const {
            return this->cntFrames;
        }

        /**
         * Gets the number of components per grid point.
         *
         * @return The number of components per grid point.
         */
        size_t GetComponents(void) const;

        /**
         * Gets the pointer to the raw data.
         *
         * @return The raw data.
         */
        inline const void *GetData(void) const {
            if (metadata != nullptr && metadata->MemLoc == RAM) {
				return this->data;
			}
			else {
				vislib::sys::Log::DefaultLog.WriteError("Trying to get volume data from VRAM: Volume data location set to RAM.");
				return nullptr;
			}
        }

        /**
         * Gets the pointer to the raw data.
         *
         * @return The raw data.
         */
        inline void *GetData(void) {
			if (metadata != nullptr && metadata->MemLoc == RAM) {
				return this->data;
			}
			else {
				vislib::sys::Log::DefaultLog.WriteError("Trying to get volume data from VRAM: Volume data location set to RAM.");
				return nullptr;
			}
        }

		inline uint32_t GetVRAMData(void) const {
			if (metadata != nullptr && metadata->MemLoc == VRAM){
				return vram_volume_name;
			}
			else {
				vislib::sys::Log::DefaultLog.WriteError("Trying to get volume data from RAM: Volume data location set to VRAM.");
				return 0;
			}
		}

        /**
         * Gets the total number of frames in the data set.
         *
         * @return The total number of frames.
         */
        size_t GetFrames(void) const;

        /**
         * Gets the size of a single frame in bytes.
         *
         * @return The size of a single frame.
         */
        inline size_t GetFrameSize(void) const {
            return this->GetVoxelSize() * this->GetVoxelsPerFrame();
        }

        /**
         * Gets the type of the grid.
         *
         * @return The type of the grid.
         */
        GridType GetGridType(void) const;

        /**
         * Gets the metadata record.
         *
         * @return The metadata record if available.
         */
        inline const Metadata *GetMetadata(void) const {
            return this->metadata;
        }

        /**
         * Gets the resolution in the specified dimension.
         *
         * @param axis The axis to retrieve the resolution for.
         *
         * @return The resolution in the specified dimension.
         *
         * @throws vislib::OutOfRangeException If 'axis' is not within [0, 3[.
         */
        const size_t GetResolution(const int axis) const;

        /**
         * Gets the length of a scalar (component) in bytes.
         *
         * @return The length of a scalar.
         */
        size_t GetScalarLength(void) const;

        /**
         * Gets the type of a scalar.
         *
         * @return The type of a scalar.
         */
        ScalarType GetScalarType(void) const;

        /**
         * Gets the distance between two slices in each of the dimensions.
         *
         * @return The slice distances.
         */
        const float *GetSliceDistances(const int axis) const;

        /**
         * Gets the number of data points in a single frame.
         */
        size_t GetVoxelsPerFrame(void) const;

        /**
         * Gets the size of a single data point in bytes.
         *
         * @return The size of a single data point.
         */
        inline size_t GetVoxelSize(void) const {
            return this->GetScalarLength() * this->GetComponents();
        }

        /**
         * Gets the voxel value relative to [min, max] from channel c.
         */
        const float GetRelativeVoxelValue(const uint32_t x, const uint32_t y, const uint32_t z, const uint32_t c = 0) const;

        /**
         * Gets the voxel value relative to [min, max] from channel c.
         */
        const float GetAbsoluteVoxelValue(
            const uint32_t x, const uint32_t y, const uint32_t z, const uint32_t c = 0) const;

        /**
         * Answer whether the given axis is uniform or has
         * this->GetResolution(axis) entries in the slice distance area.
         *
         * @param axis
         *
         * @return
         */
        bool IsUniform(const int axis) const;

        /**
         * Sets the data pointer.
         *
         * This method can be called by the caller to set a pointer to a memory
         * range which the data source should directly write the data to. The
         * caller must ensure that 'data' designates at least
         * 'cntFrames' * this->GetFrameSize() bytes.
         *
         * If the caller does not provide a pointer (ie sets nullptr), the data
         * source will allocate the memory and set the pointer.
         *
         * The object never takes ownership of the memory designated by 'data'.
         *
         * @param data
         * @param cntFrames
         */
        void SetData(void *data, const size_t cntFrames = 1) {
            this->data = data;
            this->cntFrames = cntFrames;
        }

		void SetData(uint32_t texture_name)	{
			this->vram_volume_name = texture_name;
		}

        /**
         * Update the metadata.
         *
         * @param metadata Pointer to the metadata records, which the caller
         *                 must provide as long as this call exists.
         */
        void SetMetadata(const Metadata *metadata);

        /**
         * Assignment.
         *
         * @param rhs The right hand side operator.
         *
         * @return *this.
         */
        VolumetricDataCall& operator =(const VolumetricDataCall& rhs);

    private:

        /** The base class. */
        typedef AbstractGetData3DCall Base;

        /** The functions that are provided by the call. */
        static const char *FUNCTIONS[6];

        /** The number of frames that 'data' designates. */
        size_t cntFrames;

        /** The pointer to the raw data. The call does not own this memory! */
        void *data;

		/** The texture name of the volume data if data is already located in VRAM. */
		uint32_t vram_volume_name;

        /** Pointer to the metadata descriptor of the data set. */
        const Metadata *metadata;

    };

    /** Call Descriptor.  */
    typedef factories::CallAutoDescription<VolumetricDataCall>
        VolumetricDataCallDescription;

} /* end namespace misc */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_VOLUMETRICDATACALL_H_INCLUDED */
