/*
 * VoluMetricJob.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_VOLUMETRICJOB_H_INCLUDED
#define MEGAMOLCORE_VOLUMETRICJOB_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "job/AbstractThreadedJob.h"
#include "Module.h"
#include "moldyn/MultiParticleDataCall.h"
#include "misc/LinesDataCall.h"
#include "CallTriMeshData.h"
#include "CallerSlot.h"
#include "CalleeSlot.h"
#include "param/ParamSlot.h"
#include "vislib/Cuboid.h"
#include "JobStructures.h"
#include "vislib/File.h"

namespace megamol {
namespace trisoup {
namespace volumetrics {

    /**
     * Megamol job that computes metrics about the volume occupied by a number
     * of (spherical) glyphs. Several threaded jobs are generated for a number of
     * subvolumes to speed up computation and make your machine feel the pain.
     */
    class VoluMetricJob : public core::job::AbstractThreadedJob, public core::Module {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "VoluMetricJob";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Computes volumetric metrics of a multiparticle dataset, i.e. surface and volume"
				", assuming a spacefilling spheres geometry";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return true;
        }

        /** Ctor. */
        VoluMetricJob(void);

        /** Dtor. */
        virtual ~VoluMetricJob(void);

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

        /**
         * Perform the work of a thread.
         *
         * @param userData A pointer to user data that are passed to the thread,
         *                 if it started. If the outTriDataSlot is connected, geometry is
         *                 stored inside the results (for debug displaying).
         *
         * @return The application dependent return code of the thread. This 
         *         must not be STILL_ACTIVE (259).
         */
        virtual DWORD Run(void *userData);

    private:

        /**
         * Answer whether two BorderVoxel arrays touch at least in one place.
         *
         * @param border1 the first BorderVoxel array
         * @param border2 the second BorderVoxel array
         *
         * @return whether they touch
         */
        bool doBordersTouch(vislib::Array<BorderVoxel *> &border1, vislib::Array<BorderVoxel *> &border2);

        /**
         * Provide some line geometry for rendering. Currently outputs the bounding
         * boxes of the subvolumes that are computed in parallel.
         *
         * @param caller the call where the data is inserted
         *
         * @return true if successful
         */
		bool getLineDataCallback(core::Call &caller);

        /**
         * Provide the bounding box of the line geometry.
         *
         * @param caller the call where the data is inserted
         *
         * @return true if successful
         */
		bool getLineExtentCallback(core::Call &caller);

        /**
         * Provide the collected surfaces as mesh/triangle soup with randomized colors
         * based on surface identity.
         *
         * @param caller the call where the data is inserted
         *
         * @return true if successful
         */
		bool getTriDataCallback(core::Call &caller);

        /**
         * Convenience method for generating the corner vertices of a cuboid and
         * appending them to data. offset is increased accordingly by 8 * 3 * sizeof(VoxelizerFloat).
         *
         * @see appendBoxIndices
         *
         * @param data the RawStorage where the vertices should be appended
         * @param offset Where appending starts. Is increased by 8 * 3 * sizeof(VoxelizerFloat).
         */
		void appendBox(vislib::RawStorage &data, vislib::math::Cuboid<VoxelizerFloat> &b, SIZE_T &offset);

        /**
         * Convenience method for generating the 12 indices for a wireframe of the 
         * corner vertices of a cuboid and appending them to data. offset is increased
         * accordingly by 12.
         *
         * @see appendBox
         *
         * @param data the RawStorage where the indices should be appended
         * @param offset Where appending starts. Is increased by 12.
         */
        void appendBoxIndices(vislib::RawStorage &data, unsigned int &numOffset);

        /**
         * Connects the partial geometries of completed jobs and copies the result
         * to the current backbuffer and performs a buffer swap. Increases hash to indicate this.
         * 
         * @param subJobDataList the submitted jobs
         */
        void copyMeshesToBackbuffer(vislib::Array<SubJobData*> &subJobDataList,
            vislib::Array<vislib::Array<unsigned int> > &globalSurfaceIDs,
            vislib::Array<unsigned int> &uniqueIDs);

        void generateStatistics(vislib::Array<SubJobData*> &subJobDataList,
            vislib::Array<vislib::Array<unsigned int> > &globalSurfaceIDs,
            vislib::Array<unsigned int> &uniqueIDs,
            vislib::Array<SIZE_T> &countPerID,
            vislib::Array<VoxelizerFloat> &surfPerID,
            vislib::Array<VoxelizerFloat> &volPerID);

        void outputStatistics(unsigned int frameNumber,
            vislib::Array<unsigned int> &uniqueIDs,
            vislib::Array<SIZE_T> &countPerID,
            vislib::Array<VoxelizerFloat> &surfPerID,
            vislib::Array<VoxelizerFloat> &volPerID);

        bool isSurfaceJoinableWithSubvolume(SubJobData *surfJob, int surfIdx, SubJobData *volume);

        void joinSurfaces(vislib::Array<vislib::Array<unsigned int> > &globalSurfaceIDs,
            int i, int j, int k, int l);

        core::CallerSlot getDataSlot;

        core::param::ParamSlot cellSizeRatioSlot;

		core::param::ParamSlot continueToNextFrameSlot;

        core::param::ParamSlot metricsFilenameSlot;
		
		core::param::ParamSlot showBoundingBoxesSlot;

        core::param::ParamSlot showBorderGeometrySlot;

        core::param::ParamSlot showSurfaceGeometrySlot;

        core::param::ParamSlot subVolumeResolutionSlot;

		core::param::ParamSlot radiusMultiplierSlot;

		core::param::ParamSlot resetContinueSlot;

		core::CalleeSlot outLineDataSlot;

		core::CalleeSlot outTriDataSlot;

		VoxelizerFloat MaxRad;

        VoxelizerFloat MinRad;

		/** the data hash, i.e. the front buffer frame available from the slots */
		SIZE_T hash;

		/** 
		 * which one of the two instaces of most data is the backbuffer i.e. the
		 * one that is currently being modified. the other one goes to the slots.
		 */
		char backBufferIndex;

		char meshBackBufferIndex;

        vislib::sys::File statisticsFile;

		/**
		 * lines data. debugLines[backBufferIndex][*] is writable, the other one readable.
		 * debugLines[*][0] contains the bounding boxes, [*][1] the boundaries (in future)
		 */
		megamol::core::misc::LinesDataCall::Lines debugLines[2][2];

		CallTriMeshData::Mesh debugMeshes[2];

		vislib::RawStorage bboxVertData[2];

		vislib::RawStorage bboxIdxData[2];
    };

} /* end namespace volumetrics */
} /* end namespace trisoup */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_VOLUMETRICJOB_H_INCLUDED */
