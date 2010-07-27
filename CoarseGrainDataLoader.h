/*
 * CoarseGrainDataLoader.h
 *
 * Copyright (C) 2010 by University of Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef MMPROTEINPLUGIN_COARSEGRAINDATALOADER_H_INCLUDED
#define MMPROTEINPLUGIN_COARSEGRAINDATALOADER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "Module.h"
#include "param/ParamSlot.h"
#include "CalleeSlot.h"
#include "vislib/Array.h"
#include "vislib/Vector.h"
#include "vislib/Cuboid.h"
#include "CallProteinData.h"
#include "SphereDataCall.h"

namespace megamol {
namespace protein {

    /**
	 * Data source for PDB files
	 */

	class CoarseGrainDataLoader : public megamol::core::Module
	{
	public:
		
		/** Ctor */
		CoarseGrainDataLoader(void);

		/** Dtor */
		virtual ~CoarseGrainDataLoader(void);

		/**
		 * Answer the name of this module.
		 *
		 * @return The name of this module.
		 */
		static const char *ClassName(void)  {
			return "CoarseGrainDataLoader";
		}

		/**
		 * Answer a human readable description of this module.
		 *
		 * @return A human readable description of this module.
		 */
		static const char *Description(void) {
			return "Offers protein data.";
		}

		/**
		 * Answers whether this module is available on the current system.
		 *
		 * @return 'true' if the module is available, 'false' otherwise.
		 */
		static bool IsAvailable(void) {
			return true;
		}

	protected:

		/**
		 * Implementation of 'Create'.
		 *
		 * @return 'true' on success, 'false' otherwise.
		 */
		virtual bool create(void);

        /**
         * Call callback to get the data
         *
         * @param c The calling call
         *
         * @return True on success
         */
        bool getData( core::Call& call);

        /**
         * Call callback to get the extent of the data
         *
         * @param c The calling call
         *
         * @return True on success
         */
        bool getExtent( core::Call& call);

		/**
		 * Implementation of 'Release'.
		 */
		virtual void release(void);

	private:

        /**
         * Loads a PDB file.
         *
         * @param filename The path to the file to load.
         */
        void loadFile( const vislib::TString& filename);

        /**
         * Parse one entry.
         *
         * @param entry The atom entry string.
         * @param idx   The number of the current atom.
         * @param frame The number of the current frame.
         */
        void parseEntry( vislib::StringA &entry, unsigned int idx, unsigned int frame);

        // -------------------- variables --------------------

        /** The file name slot */
        core::param::ParamSlot filenameSlot;
        /** The data callee slot */
        core::CalleeSlot dataOutSlot;

        /** The number of frames */
        unsigned int frameCount;

        /** The number of spheres per frame */
        unsigned int sphereCount;

        /** The data */
        vislib::Array<vislib::Array<float> > data;

        /** The sphere charge */
        vislib::Array<vislib::Array<float> > sphereCharge;
        /** The maximum charge in the trajectory. */
        float maxCharge;
        /** The minimum charge in the trajectory. */
        float minCharge;

        /** The sphere colors */
        vislib::Array<unsigned char> sphereColor;

        /** The sphere types */
        vislib::Array<unsigned int> sphereType;

        /** The bounding box */
        vislib::math::Cuboid<float> bbox;

        /** The data hash */
        SIZE_T datahash;

	};


} /* end namespace protein */
} /* end namespace megamol */

#endif // MMPROTEINPLUGIN_COARSEGRAINDATALOADER_H_INCLUDED
