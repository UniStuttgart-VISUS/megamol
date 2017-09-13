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
#include "vislib/Array.h"
#include "vislib/math/Plane.h"

#include <vector>

extern "C" void clearAll(void);
extern "C" void getPositions(float * h_atomPositions, unsigned int numPositions);
extern "C" void performTimestep(float timestepSize, bool forceToCenter, float forceStrength);
extern "C" void transferAtomData(float * h_atomPositions, unsigned int numPositions, unsigned int * h_cAlphaIndices, unsigned int numCAlphas);
extern "C" void transferPlane(vislib::math::Plane<float>& thePlane);
extern "C" void transferSpringData(const float * h_atomPositions, unsigned int numPositions, const unsigned int * h_hBondIndices, unsigned int numBonds, 
	unsigned int * h_cAlphaIndices, unsigned int numCAlphas, unsigned int * h_oIndices, unsigned int numOs, float conFriction, float conConstant, 
	float hFriction, float hConstant, const unsigned int * moleculeStarts, unsigned int numMolecules, float cutoffDistance, float strengthFactor);

namespace megamol {
namespace protein_cuda {

	class SecStructFlattener : public megamol::core::Module {
	public:

		/**
         *	Answer the name of this module.
         *	
         *	@return The name of this module.
         */
		static const char *ClassName(void) {
			return "SecStructFlattener";
		}

		/**
         *	Answer a human readable description of this module.
         *
         *	@return A human readable description of this module.
         */
		static const char *Description(void) {
			return "Flattens the secondary structure of a protein into a 2D plane";
		}

		/**
         *	Answers whether this module is available on the current system.
         *	
         *	@return 'true' if the module is available, 'false' otherwise.
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
         *	Implementation of 'Create'.
         *	
         *	@return 'true' on success, 'false' otherwise.
         */
		virtual bool create(void);

		/**
         *	Implementation of 'release'.
         */
		virtual void release(void);

		/**
         *	Call for get data.
         */
		bool getData(megamol::core::Call& call);

		/**
         *	Call for get extent.
         */
		bool getExtent(megamol::core::Call& call);

		/**
		 *	Call for get plane data.
		 */
		bool getPlaneData(megamol::core::Call& call);

		/**
		 *	Call for get plane extent.
		 */
		bool getPlaneExtent(megamol::core::Call& call);

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
		 *
		 *	@param tranferPositions Should the atom positions be tranferred to the cuda kernel?
		 */
		void flatten(bool transferPositions = false);

		/**
		 *	Runs the simulation based on the tranferred parameters.
		 */
		void runSimulation(void);

		/**
		 *	Callback function for the animation play button.
		 *
		 *	@param p The button parameter
		 *	@return true
		 */
		bool onPlayToggleButton(megamol::core::param::ParamSlot& p);

		/**
		 *	Callback function for the single timestep button.
		 *
		 *	@param p The button parameter
		 *	@return true
		 */
		bool onSingleStepButton(megamol::core::param::ParamSlot& p);

		/**
		 *	Callback function for the reset button.
		 *
		 *	@param p The button parameter
		 *	@return true
		 */
		bool onResetButton(megamol::core::param::ParamSlot& p);

		/**
		 *	Computes the three main directions of the c alpha atoms
		 */
		void computeMainDirectionPCA(void);

		/** data caller slot */
		megamol::core::CallerSlot getDataSlot;

		/** slot for outgoing data */
		megamol::core::CalleeSlot dataOutSlot;

		/** slot for outgoing plane data */
		megamol::core::CalleeSlot planeOutSlot;

		/** toggles the play of the animation */
		megamol::core::param::ParamSlot playParam;

		/** button that toggles the play of the animation */
		megamol::core::param::ParamSlot playButtonParam;

		/** button that triggers the computation of a single timestep */
		megamol::core::param::ParamSlot singleStepButtonParam;

		/** the flat plane mode */
		megamol::core::param::ParamSlot flatPlaneMode;

		/** the normal of the arbitrary plane */
		megamol::core::param::ParamSlot arbPlaneNormalParam;
												
		/** the origin of the arbitrary plane */
		megamol::core::param::ParamSlot arbPlaneCenterParam;

		/** Preserve the offset of the oxygen atoms relative to the c alphas? */
		megamol::core::param::ParamSlot oxygenOffsetParam;

		/** Duration of a single timestep */
		megamol::core::param::ParamSlot timestepParam;

		/** Maximal number of performed timesteps */
		megamol::core::param::ParamSlot maxTimestepParam;

		/** Number of performed timesteps per frame */
		megamol::core::param::ParamSlot timestepsPerFrameParam;

		/** The spring constant for the atom connection springs */
		megamol::core::param::ParamSlot connectionSpringConstantParam;

		/** The spring constant for the h bond springs */
		megamol::core::param::ParamSlot hbondSpringConstantParam;

		/** The friction parameter for the atom connection springs */
		megamol::core::param::ParamSlot connectionFrictionParam;

		/** The friction parameter for the h bond springs */
		megamol::core::param::ParamSlot hbondFrictionParam;

		/** The cutoff distance for the repelling forces */
		megamol::core::param::ParamSlot repellingForceCutoffDistanceParam;

		/** Factor controlling the strength of the repelling forces */
		megamol::core::param::ParamSlot repellingForceStrengthFactor;

		/** Flag activating a force to the center of the bounding box */
		megamol::core::param::ParamSlot forceToCenterParam;

		/** The strength of the force to the center of the bounding box */
		megamol::core::param::ParamSlot forceToCenterStrengthParam;

		/** The reset button */
		megamol::core::param::ParamSlot resetButtonParam;

		/** The current atom positions */
		float * atomPositions;

		/** The size of the atom positions array */
		unsigned int atomPositionsSize;

		/** The bounding box of the data */
		vislib::math::Cuboid<float> boundingBox;

		/** The indices of the c alpha atoms */
		std::vector<unsigned int> cAlphaIndices;

		/** The indices of the oxygen atoms */
		std::vector<unsigned int> oIndices;

		/** The last hash of the used data set */
		SIZE_T lastHash;

		/** The current hash of the data emitted by this module */
		SIZE_T myHash;

		/** The offset to the hash from the data set hash */
		SIZE_T hashOffset;

		/** The current plane hash */
		SIZE_T planeHash;

		/** The lastly used plane mode */
		FlatPlane lastPlaneMode;

		/** The main directions of the c alpha atoms of the data set, ordered by significance */
		std::vector<vislib::math::Vector<float, 3>> mainDirections;

		/** Indicator for the first frame */
		bool firstFrame;

		/** The distance vectors from the c alpha atoms to the corresponding oxygen atoms */
		std::vector<vislib::math::Vector<float, 3>> oxygenOffsets;

		/** The currently used projection plane */
		vislib::math::Plane<float> currentPlane;

		/** The index of the current timestep */
		SIZE_T currentTimestep;

		/** Should a single timestep be performed? */
		bool oneStep;

		/** Should a reset of the dataset be forced? */
		bool forceReset;
	};

} /* end namespace protein_cuda */
} /* end namespace megamol */

#endif // MMPROTEINCUDAPLUGIN_SECSTRUCTFLATTENER_H_INCLUDED