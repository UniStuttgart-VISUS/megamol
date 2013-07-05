/*
 * SolventDataGenerator.h
 *
 * Copyright (C) 2011 by University of Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef MMPROTEINPLUGIN_SolventDataGenerator_H_INCLUDED
#define MMPROTEINPLUGIN_SolventDataGenerator_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "param/ParamSlot.h"
#include "CalleeSlot.h"
#include "CallerSlot.h"
#include "GridNeighbourFinder.h"
#include "vislib/Array.h"
#include "vislib/Vector.h"
#include "vislib/Cuboid.h"
#include "MolecularDataCall.h"
#include "Stride.h"
#include "view/AnimDataModule.h"
#include <fstream>

namespace megamol {
namespace protein {


    /**
     * generator for hydrogent bounds etc ...
	 * this class can be put in place between PDBLoader and a molecule renderer (SolventVolumeRenderer for example)...
     */

    class SolventDataGenerator : public megamol::core::/*view::AnimData*/Module {
    public:

        /** Ctor */
        SolventDataGenerator(void);

        /** Dtor */
        virtual ~SolventDataGenerator(void);

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void)  {
            return "SolventDataGenerator";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Special molecule data preprocessing stepts (from PDB data).";
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
        bool getData(core::Call& call);

        bool getExtent(core::Call& call);

		bool dataChanged(core::Call& call);

		/**
		 * for now just calculate the arithmetic average positions of each atom ...
		 */
		void calcSpatialProbabilities(MolecularDataCall *src, MolecularDataCall *dst);

		/**
		 * Get the hydrogen bonds for the current frameID of 'dataSource' and store the result in 'dataTarget'.
		 * The hydrogen-bonds may be already precomputed (in core or file-IO) so this function won't take much time.
		 */
		bool getHBonds(MolecularDataCall *dataTarget, MolecularDataCall *dataSource);

		/**
		 * calculate hydrogen-bonds with a neighbor-search of possible donors/acceptors and check for connected hydrogens ...
		 */
		void calcHydroBondsForCurFrame(MolecularDataCall *data, const float *atomPositions, int *atomHydroBondsIndicesPtr);
		inline void calcHydroBondsForCurFrame(MolecularDataCall *data, int *atomHydroBondsIndicesPtr) {
			calcHydroBondsForCurFrame(data, data->AtomPositions(), atomHydroBondsIndicesPtr);
		}

		/**
		 * create hydrogen-bond statistics for the polymer atoms ...
		 */
		bool calcHydrogenBondStatistics(MolecularDataCall *dataTarget, MolecularDataCall *dataSource);

		/**
         * Implementation of 'Release'.
         */
        virtual void release(void);

	private:

        /**********************************************************************
         * variables
         **********************************************************************/

        /** caller slot */
        megamol::core::CallerSlot molDataInputCallerSlot;

        /** The data callee slot */
        megamol::core::CalleeSlot dataOutSlot;

		/** Distance for hydrogen bonds */
		megamol::core::param::ParamSlot hBondDistance;
		megamol::core::param::ParamSlot hBondDonorAcceptorDistance;
		megamol::core::param::ParamSlot hBondDonorAcceptorAngle;
		//megamol::core::param::ParamSlot hBondDataFile;
		megamol::core::param::ParamSlot showMiddlePositions;

		/** temporary variable to store a set of atom positions */
		vislib::Array<float> middleAtomPos;
		vislib::Array<int> middleAtomPosHBonds;

		/** our grid based neighbour finder ... */
		GridNeighbourFinder<float> neighbourFinder;

		/** temporary variable to store the neighbour indices for the hydrogen-bound search ...*/
		vislib::Array<unsigned int> *neighbourIndices;
		//vislib::Array<unsigned int> *neighbHydrogenIndices;
		/** store hydrogen connections per atom ... */
		vislib::Array<int> hydrogenConnections;
		vislib::Array<int> donorAcceptors;
		vislib::Array<unsigned int> hydrogenBondStatistics;
		enum { MAX_HYDROGENS_PER_ATOM = 4 };
		//enum { DONOR_ACCEPTOR_TYPE_COUNT = 2 /* only 'O' and 'N' can be donor/acceptor*/};
		int maxOMPThreads;

		/** array to check atoms already connected ... */
		vislib::Array<int> reverseConnection;

		/* store 2 hydrogen bounds in core so interpolating between two frames can be done without file-access */
		enum { HYDROGEN_BOND_IN_CORE = 3/*20*//*2*/ };

		/** store hydrogen bounds ...*/
		vislib::Array<int> atomHydroBondsIndices[HYDROGEN_BOND_IN_CORE];
		int curHBondFrame[HYDROGEN_BOND_IN_CORE];

		inline bool validHydrogenBond(int donorIdx, int hydrogenIdx, int acceptorIdx, const float *atomPositions, float angle) const {
			vislib::math::ShallowPoint</*const*/ float,3> donorPos(const_cast<float*>(atomPositions)+donorIdx*3);
			vislib::math::ShallowPoint</*const*/ float,3> acceptorPos(const_cast<float*>(atomPositions)+acceptorIdx*3);
			vislib::math::ShallowPoint</*const*/ float,3> hydrogenPos(const_cast<float*>(atomPositions)+hydrogenIdx*3);
			//vislib::math::Vector<float,3> tmp(hydrogenPos-donorPos);
			//vislib::math::Vector<float,3> tmp2(acceptorPos-donorPos);
			return (hydrogenPos-donorPos).Angle(acceptorPos-donorPos) <= angle;
		}
	};


} /* end namespace protein */
} /* end namespace megamol */

#endif // MMPROTEINPLUGIN_SolventDataGenerator_H_INCLUDED
