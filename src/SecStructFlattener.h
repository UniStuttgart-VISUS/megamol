/*
 *	SecStructFlattener.h
 *
 *	Copyright (C) 2016 by Universitaet Stuttgart (VISUS).
 *	All rights reserved
 */

#ifndef MMPROTEINCUDAPLUGIN_SECSTRUCTFLATTENER_H_INCLUDED
#define MMPROTEINCUDAPLUGIN_SECSTRUCTFLATTENER_H_INCLUDED

#include "mmcore/Module.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/CalleeSlot.h"
#include "protein_calls/MolecularDataCall.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/math/Cuboid.h"

namespace megamol {
namespace protein_cuda {

	class SecStructFlattener : public megamol::core::Module {
	public:

		/**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
		static const char *ClassName(void) {
			return "SecStructFlattener";
		}

		/**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
		static const char *Description(void) {
			return "Flattens the secondary structure of a protein into a 2D plane";
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
		SecStructFlattener(void);

		/** Dtor. */
		virtual ~SecStructFlattener(void);

	protected:

		/**
         * Implementation of 'Create'.
         *
         * @return 'true' on success, 'false' otherwise.
         */
		virtual bool create(void);

		/**
         * Implementation of 'release'.
         */
		virtual void release(void);

		/**
         * Call for get data.
         */
		bool getData(megamol::core::Call& call);

		/**
         * Call for get extent.
         */
		bool getExtent(megamol::core::Call& call);

	private:

		/**
		 *	Enum for the plane the protein should be flattened to.
		 */
		enum FlatPlane{
			XY_PLANE = 0,
			XZ_PLANE = 1,
			YZ_PLANE = 2,
			LEAST_COMMON = 3,
			ARBITRARY = 4
		};

		/**
		 *	Returns the name of the plane the protein gets flattened to.
		 *
		 *	@param fp The flat plane
		 *	@return The name of the flat plane
		 */
		std::string getFlatPlaneName(FlatPlane fp);

		/**
		 *	Returns the flat plane with the given index.
		 *	
		 *	@param idx The index of the flat plane.
		 *	@return The flat plane with the given index.
		 */
		FlatPlane getFlatPlaneByIndex(unsigned int idx);

		/**
		 *	Returns the number of flat plane modes.
		 *
		 *	@return The number of flat plane modes.
		 */
		int getFlatPlaneModeNumber(void);

		/**
		 *	Flattens the secondary structure of the protein progressively.
		 */
		void flatten(void);

		/**
		 *	Callback function for the animation play button.
		 *
		 *	@param p The button parameter
		 *	@return true
		 */
		bool onPlayToggleButton(megamol::core::param::ParamSlot& p);

		/** data caller slot */
		megamol::core::CallerSlot getDataSlot;

		/** slot for outgoing data */
		megamol::core::CalleeSlot dataOutSlot;

		/** toggles the play of the animation */
		megamol::core::param::ParamSlot playParam;

		/** button that toggles the play of the animation */
		megamol::core::param::ParamSlot playButtonParam;

		/** the flat plane mode */
		megamol::core::param::ParamSlot flatPlaneMode;

		/** The current atom positions */
		float * atomPositions;

		/** The size of the atom positions array */
		unsigned int atomPositionsSize;

		/** The bounding box of the data */
		vislib::math::Cuboid<float> boundingBox;
	};

} /* end namespace protein_cuda */
} /* end namespace megamol */

#endif // MMPROTEINCUDAPLUGIN_SECSTRUCTFLATTENER_H_INCLUDED