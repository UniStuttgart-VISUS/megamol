//
// ProteinExploder.h
//
// Copyright (C) 2016 by University of Stuttgart (VISUS).
// All rights reserved.
//

#ifndef MMPROTEINPLUGIN_PROTEINEXPLODER_H_INCLUDED
#define MMPROTEINPLUGIN_PROTEINEXPLODER_H_INCLUDED

#include "mmcore/view/AnimDataModule.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/CalleeSlot.h"
#include "protein_calls/MolecularDataCall.h"
#include "mmcore/param/ParamSlot.h"

namespace megamol {
namespace protein {
	
	class ProteinExploder : public megamol::core::view::AnimDataModule 
	{
	public:

		/**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
		static const char *ClassName(void) {
			return "ProteinExploder";
		}

		/**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
		static const char *Description(void) {
			return "Pulls proteins apart to have a better view on single components.";
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
		ProteinExploder(void);

		/** Dtor. */
		virtual ~ProteinExploder(void);

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
         * Creates a frame to be used in the frame cache. This method will be
         * called from within 'initFrameCache'.
         *
         * @return The newly created frame object.
         */
        virtual Frame* constructFrame(void) const;

        /**
         * Loads one frame of the data set into the given 'frame' object. This
         * method may be invoked from another thread. You must take
         * precausions in case you need synchronised access to shared
         * ressources.
         *
         * @param frame The frame to be loaded.
         * @param idx The index of the frame to be loaded.
         */
        virtual void loadFrame(Frame *frame, unsigned int idx);

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
         * Storage of frame data
         */
        class Frame : public megamol::core::view::AnimDataModule::Frame {
        public:

            /** Ctor */
            Frame(megamol::core::view::AnimDataModule& owner);

            /** Dtor */
            virtual ~Frame(void);

            /**
             * Set the frame Index.
             *
             * @param idx the index
             */
            void setFrameIdx(int idx);

            /**
             * Test for equality
             *
             * @param rhs The right hand side operand
             *
             * @return true if this and rhs are equal
             */
            bool operator==(const Frame& rhs);

            /**
             * Set the atom count.
             *
             * @param atomCnt The atom count
             */
            inline void SetAtomCount( unsigned int atomCnt) {
                this->atomCount = atomCnt;
                this->atomPosition.SetCount( atomCnt*3);
                this->bfactor.SetCount( atomCnt);
                this->charge.SetCount( atomCnt);
                this->occupancy.SetCount( atomCnt);
            }

            /**
             * Get the atom count.
             *
             * @return The atom count.
             */
            inline unsigned int AtomCount() const { return this->atomCount; }

            /**
             * Assign a position to the array of positions.
             */
            bool SetAtomPosition( unsigned int idx, float x, float y, float z);

            /**
             * Assign a bfactor to the array of bfactors.
             */
            bool SetAtomBFactor( unsigned int idx, float val);

            /**
             * Assign a charge to the array of charges.
             */
            bool SetAtomCharge( unsigned int idx, float val);

            /**
             * Assign a occupancy to the array of occupancies.
             */
            bool SetAtomOccupancy( unsigned int idx, float val);

            /**
             * Set the b-factor range.
             *
             * @param min    The minimum b-factor.
             * @param max    The maximum b-factor.
             */
            void SetBFactorRange( float min, float max) {
                this->minBFactor = min; this->maxBFactor = max; }

            /**
             * Set the minimum b-factor.
             *
             * @param min    The minimum b-factor.
             */
            void SetMinBFactor( float min) { this->minBFactor = min; }

            /**
             * Set the maximum b-factor.
             *
             * @param max    The maximum b-factor.
             */
            void SetMaxBFactor( float max) { this->maxBFactor = max; }

            /**
             * Set the charge range.
             *
             * @param min    The minimum charge.
             * @param max    The maximum charge.
             */
            void SetChargeRange( float min, float max) {
                this->minCharge = min; this->maxCharge = max; }

            /**
             * Set the minimum charge.
             *
             * @param min    The minimum charge.
             */
            void SetMinCharge( float min) { this->minCharge = min; }

            /**
             * Set the maximum charge.
             *
             * @param max    The maximum charge.
             */
            void SetMaxCharge( float max) { this->maxCharge = max; }

            /**
             * Set the occupancy range.
             *
             * @param min    The minimum occupancy.
             * @param max    The maximum occupancy.
             */
            void SetOccupancyRange( float min, float max) {
                this->minOccupancy = min; this->maxOccupancy = max; }

            /**
             * Set the minimum occupancy.
             *
             * @param min    The minimum occupancy.
             */
            void SetMinOccupancy( float min) { this->minOccupancy = min; }

            /**
             * Set the maximum occupancy.
             *
             * @param max    The maximum occupancy.
             */
            void SetMaxOccupancy( float max) { this->maxOccupancy = max; }

            /**
             * Get a reference to the array of atom positions.
             *
             * @return The atom position array.
             */
            const float* AtomPositions() { return this->atomPosition.PeekElements(); }

            /**
             * Get a reference to the array of atom b-factors.
             *
             * @return The atom b-factor array.
             */
            const float* AtomBFactor() { return this->bfactor.PeekElements(); }

            /**
             * Get a reference to the array of atom charges.
             *
             * @return The atom charge array.
             */
            const float* AtomCharge() { return this->charge.PeekElements(); }

            /**
             * Get a reference to the array of atom occupancies.
             *
             * @return The atom occupancy array.
             */
            const float* AtomOccupancy() { return this->occupancy.PeekElements(); }

            /**
             * Get the maximum b-factor of this frame.
             *
             * @return The maximum b-factor.
             */
            float MaxBFactor() const { return this->maxBFactor; }

            /**
             * Get the minimum b-factor of this frame.
             *
             * @return The minimum b-factor.
             */
            float MinBFactor() const { return this->minBFactor; }

            /**
             * Get the maximum b-factor of this frame.
             *
             * @return The maximum b-factor.
             */
            float MaxCharge() const { return this->maxCharge; }

            /**
             * Get the minimum charge of this frame.
             *
             * @return The minimum charge.
             */
            float MinCharge() const { return this->minCharge; }

            /**
             * Get the maximum occupancy of this frame.
             *
             * @return The maximum occupancy.
             */
            float MaxOccupancy() const { return this->maxOccupancy; }

            /**
             * Get the minimum occupancy of this frame.
             *
             * @return The minimum occupancy.
             */
            float MinOccupancy() const { return this->minOccupancy; }

        private:
            /** The atom count */
            unsigned int atomCount;

            /** The atom positions */
            vislib::Array<float> atomPosition;

            /** The atom b-factors */
            vislib::Array<float> bfactor;

            /** The atom charges */
            vislib::Array<float> charge;

            /** The atom occupancy */
            vislib::Array<float> occupancy;

            /** The maximum b-factor */
            float maxBFactor;
            /** The minimum b-factor */
            float minBFactor;

            /** The maximum carge */
            float maxCharge;
            /** The minimum charge */
            float minCharge;

            /** The maximum occupancy */
            float maxOccupancy;
            /** The minimum occupancy */
            float minOccupancy;

        };

		/**
		 *	Enum for the mode of explosion
		 */
		enum ExplosionMode {
			SPHERICAL_MIDDLE = 0,
			SPHERICAL_MASS = 1,
			MAIN_DIRECTION = 2,
			MAIN_DIRECTION_CIRCULAR = 3
		};

		/**
		 *	Returns the name of the given explosion mode
		 *
		 *	@param mode The explosion mode
		 *	@return The name of the explosion mode as string
		 */
		std::string getModeName(ExplosionMode mode);

		/*
		 *	Returns the explosion mode with the given index
		 *	
		 *	@param idx The index of the explosion mode
		 *	@return The explosion mode with the given index
		 */
		ExplosionMode getModeByIndex(unsigned int idx);

		/**
		 *	Returns the number of explosion modes.
		 */
		int getModeNumber();

		/**
		 *	Displaces the atom positions of a molecule dependant from the given parameters
		 *
		 *	@param call The call containing the molecule data
		 *	@param mode The explosion mode
		 *	@param minDistance The minimal distance between two exploded parts
		 */
		void explodeMolecule(megamol::protein_calls::MolecularDataCall& call, ExplosionMode mode, float minDistance);

		/** data caller slot */
		megamol::core::CallerSlot getDataSlot;

		/** slot for outgoing data */
		megamol::core::CalleeSlot dataOutSlot;

		/** minimal distance between two exploded components */
		megamol::core::param::ParamSlot minDistanceParam;

		/** The explosion mode */
		megamol::core::param::ParamSlot explosionModeParam;

		/** The current atom positions */
		float * atomPositions;

		/** The size of the atom positions array */
		unsigned int atomPositionsSize;

		/** The current bounding box */
		vislib::math::Cuboid<float> currentBoundingBox;
	};
} /* namespace protein */
} /* namespace megamol */

#endif // MMPROTEINPLUGIN_PROTEINEXPLODER_H_INCLUDED