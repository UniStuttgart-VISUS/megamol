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

		/** The last known data hash of the incoming data */
		SIZE_T lastDataHash;
		
		/** The offset from the last known data hash */
		SIZE_T dataHashOffset;
	};

} /* end namespace protein */
} /* end namespace megamol */

#endif /* MMPROTEINPLUGIN_HYDROBONDGENERATOR_H_INCLUDED */