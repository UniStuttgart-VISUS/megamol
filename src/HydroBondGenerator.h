/*
 *	HydroBondGenerator.h
 *	
 *	Copyright (C) 2016 by University of Stuttgart (VISUS).
 *	All rights reserved.
 */

#ifndef MMPROTEINPLUGIN_HYDROBONDGENERATOR_H_INCLUDED
#define MMPROTEINPLUGIN_HYDROBONDGENERATOR_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/Module.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "protein_calls/MolecularDataCall.h"

namespace megamol {
namespace protein {

	class HydroBondGenerator : public core::Module {
	public:

		/** Ctor */
		HydroBondGenerator(void);

		/** Dtor */
		virtual ~HydroBondGenerator(void);

		/**
         *	Answer the name of this module.
         *	
         *	@return The name of this module.
         */
        static const char *ClassName(void)  {
            return "HydroBondGenerator";
        }

        /**
         *	Answer a human readable description of this module.
         *	
         *	@return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Computes hydrogen bonds of given molecules.";
        }

        /**
         *	Answers whether this module is available on the current system.
         *	
         *	@return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return true;
        }

	protected:
		/**
         *	Implementation of 'Create'.
         *	
         *	@return 'true' on success, 'false' otherwise.
         */
        virtual bool create(void);

		/**
         *	Implementation of 'Release'.
         */
        virtual void release(void);

        /**
         *	Call callback to get the data
         *	
         *	@param c The calling call
         *	@return True on success
         */
        bool getData(core::Call& call);

		/**
         *	Call callback to get the extents
         *	
         *	@param c The calling call
         *	@return True on success
         */
        bool getExtent(core::Call& call);

	private:

		/**
		 *	Struct representing a hydrogen bond
		 */
		struct HBond {

			/** Index of the donor atom */
			unsigned int donorIdx;
			/** Index of the acceptor atom */
			unsigned int acceptorIdx;
			/** Index of the hydrogen atom */
			unsigned int hydrogenIdx;
			/** The angle of the connection */
			float angle;

			/** Ctor. */
			HBond() : donorIdx(0), acceptorIdx(0), hydrogenIdx(0), angle(0.0f) {}

			/**
			 *	Constructor for a hydrogen bond.
			 *
			 *	@param donorIdx The index of the donor atom.
			 *	@param acceptorIdx The index of the acceptor atom.
			 *	@param hydrogenIdx The index of the hydrogen atom.
			 *	@param angle The angle of the connection.
			 */
			HBond(unsigned int donorIdx, unsigned int acceptorIdx, unsigned int hydrogenIdx, float angle = -1.0f) : 
				donorIdx(donorIdx), acceptorIdx(acceptorIdx), hydrogenIdx(hydrogenIdx), angle(angle) {}

			/**
			 *	Overload of the comparison operator
			 *
			 *	@param rhs The right hand side of the comparison
			 *	@return True, when this is smaller than the right hand side, false otherwise.
			 */
			bool operator<(const HBond& rhs) {
				if (this->donorIdx == rhs.donorIdx)
					return this->angle < rhs.angle;
				else
					return this->donorIdx < rhs.donorIdx;
			}

			/**
			 *	Prints a representation of this struct to the console
			 */
			void print() {
				printf("H-Bond from %u to %u over %u with angle %f\n", donorIdx, acceptorIdx, hydrogenIdx, angle);
			}
		};

		/**
		 *	Computes the H Bonds for the given MolecularDataCall
		 *
		 *	@param mdc The given call.
		 */
		void computeHBonds(protein_calls::MolecularDataCall& mdc);

		/**
		 *	Post process the computed hydrogen bonds by deleting unneccessary ones.
		 *
		 *	@param mdc The call storing all relevant molecular information.
		 */
		void postProcessHBonds(protein_calls::MolecularDataCall& mdc);

		/**
		 *	Fills the secondary structure vector with information about the secondary structure of every atom.
		 *
		 *	@param mdc The call storing all relevant molecular information.
		 */
		void fillSecStructVector(protein_calls::MolecularDataCall& mdc);

		/**
		 *	Searches in the connections array for the connections starting at a certain atom.
		 *	It is assumed that the connections array is sorted.
		 *	
		 *	@param atomIdx The atom index to search the connections for.
		 *	@param firstIdx OUT: The index of the first connection.
		 *	@param lastIdx OUT: The index of the last connection.
		 *	@return True, when there are connections with atomIdx as start, false otherwise.
		 */
		bool findConnections(unsigned int atomIdx, unsigned int & firstIdx, unsigned int & lastIdx);

		/**
		 *	Computes whether a given hydrogen bond is valid.
		 *
		 *	@param donorIndex The index of the donor of the hydrogen bond.
		 *	@param accptorIndex The index of the acceptor of the hydrogen bond.
		 *	@param hydrogenIndex The index of the hydrogen atom of the bond.
		 *	@param mdc The MolecularDataCall containing the data.
		 *	@param angle OUT: The angle of the hydrogen bond.
		 */
		bool isValidHBond(unsigned int donorIndex, unsigned int acceptorIndex, unsigned int hydrogenIndex, protein_calls::MolecularDataCall& mdc, float & angle);

		/** caller slot */
		core::CallerSlot inDataSlot;

		/** callee slot */
		core::CalleeSlot outDataSlot;

		/** Maximal distance for hydrogen bonds */
		core::param::ParamSlot hBondDistance;

		/** Maximal distance between donor and acceptor of a hydrogen bond */
		core::param::ParamSlot hBondDonorAcceptorDistance;

		/** Maximal angle between donor-acceptor and donor-hydrogen */
		core::param::ParamSlot hBondDonorAcceptorAngle;

		/** Should the H-Bonds of the alpha helices be computed? */
		core::param::ParamSlot alphaHelixHBonds;

		/** Should the H-Bonds of the beta sheets be computed? */
		core::param::ParamSlot betaSheetHBonds;

		/** Should the rest of the H-Bonds be computed */
		core::param::ParamSlot otherHBonds;

		/** Maximal number of hydrogen bonds per atom */
		core::param::ParamSlot maxHBondsPerAtom;

		/** Should the H-Bonds be faked as bonds between two c-alphas? */
		core::param::ParamSlot cAlphaHBonds;

		/** The last known data hash of the incoming data */
		SIZE_T lastDataHash;
		
		/** The offset from the last known data hash */
		SIZE_T dataHashOffset;

		/** Vector containing all hydrogen bonds found */
		std::vector<HBond> hydrogenBonds;

		/** Number of H-bonds per atom */
		std::vector<unsigned int> hBondStatistics;

		/** Indices of the connected H-bond H-atom if the atom is connected within a H-bond. -1 otherwise */
		std::vector<int> hBondIndices;

		/** Sorted atom connection array for fast search */
		std::vector<std::pair<unsigned int, unsigned int>> connections;

		/** The starting indices of the connections belonging to each atom */
		std::vector<int> connectionStart;

		/** The number of connections of each atom */
		std::vector<unsigned int> numConnections;

		/** The secondary structure ID per atom */
		std::vector<protein_calls::MolecularDataCall::SecStructure::ElementType> secStructPerAtom;

		/** The c alpha indices per atom */
		std::vector<unsigned int> cAlphaIndicesPerAtom;
	};

} /* end namespace protein */
} /* end namespace megamol */

#endif /* MMPROTEINPLUGIN_HYDROBONDGENERATOR_H_INCLUDED */