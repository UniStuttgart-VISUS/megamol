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
//#include "view/Renderer3DModule.h"
//#include "Call.h"
#include "CallerSlot.h"
#include "CalleeSlot.h"
#include "param/ParamSlot.h"
#include "vislib/Cuboid.h"
//#include "vislib/memutils.h"


namespace megamol {
namespace trisoup {

    /**
     * TODO: Document
     */
    class VoluMetricJob : public core::job::AbstractThreadedJob, public core::Module {
    public:

		/**
		 *
		 */
		struct SubJobResult {

			// waer doch echt gut, wenn der schon komplette schalen getrennt rausgeben wuerde oder?

			// schon gleich in nem format, das die trisoup mag.

			// also einfach ueber das ganze volumen iterieren:
			// oberflaeche gefunden?
			//   ja: neu?
			//     ja: grow + tag cells
			//     nein: ignore

			float *vertices;
			float *normals;
			float surface;
		};

		/**
		 * Structure that hold all parameters so a single thread can work
		 * on his subset of the total volume occupied by a particle list
		 */
		struct SubJobData {
			/**
			 * Volume to work on and rasterize the data of
			 */
			vislib::math::Cuboid<float> Bounds;

			/**
			 * Edge length of a voxel (set in accordance to the radii of the contained particles)
			 */
			float CellSize;

			/**
			 * Datacall that can answer all particles not yet clipped against the subjob bounds
			 */
			megamol::core::moldyn::MultiParticleDataCall *Datacall;

			/**
			 * Here the Job should store its results. It is pre-allocated by the scheduling thread!
			 */
			SubJobResult *Result;
		};

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

        VoluMetricJob(void);

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
         *                 if it started.
         *
         * @return The application dependent return code of the thread. This 
         *         must not be STILL_ACTIVE (259).
         */
        virtual DWORD Run(void *userData);

    private:

		bool getLineDataCallback(core::Call &caller);

		bool getLineExtentCallback(core::Call &caller);

		void appendBox(vislib::RawStorage &data, vislib::math::Cuboid<float> &b, SIZE_T &offset);

		void appendBoxIndices(vislib::RawStorage &data, unsigned int &numOffset);

        core::CallerSlot getDataSlot;

        core::param::ParamSlot metricsFilenameSlot;

		core::CalleeSlot outLineDataSlot;

		float MaxRad;

		/** the data hash, i.e. the front buffer frame available from the slots */
		SIZE_T hash;

		/** 
		 * which one of the two instaces of most data is the backbuffer i.e. the
		 * one that is currently being modified. the other one goes to the slots.
		 */
		char backBufferIndex;

		/**
		 * lines data. debugLines[backBufferIndex][*] is writable, the other one readable.
		 * debugLines[*][0] contains the bounding boxes, [*][1] the boundaries (in future)
		 */
		megamol::core::misc::LinesDataCall::Lines debugLines[2][2];

		vislib::RawStorage bboxVertData[2];

		vislib::RawStorage bboxIdxData[2];
    };

} /* end namespace trisoup */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_VOLUMETRICJOB_H_INCLUDED */
