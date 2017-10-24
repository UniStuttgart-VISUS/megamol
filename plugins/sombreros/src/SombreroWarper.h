/*
 * SombreroWarper.h
 * Copyright (C) 2006-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#ifndef MMSOMBREROSPLUGIN_SOMBREROWARPER_H_INCLUDED
#define MMSOMBREROSPLUGIN_SOMBREROWARPER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/Module.h"
#include "mmcore/Call.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/param/ParamSlot.h"

#include "mmstd_trisoup/CallTriMeshData.h"

namespace megamol {
namespace sombreros {

	class SombreroWarper : public core::Module {
	public:

		/**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "SombreroWarper";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Module that warps a given mesh to resemble the shape of a sombrero.";
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
		SombreroWarper(void);

		/** Dtor. */
		virtual ~SombreroWarper(void);

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

		/** The lastly received data hash */
		SIZE_T lastDataHash;

		/** The offset to the lastly received hash */
		SIZE_T hashOffset;

		/** Slot for the mesh input. */
		core::CallerSlot meshInSlot;

		/** Slot for the ouput of the cut mesh */
		core::CalleeSlot warpedMeshOutSlot;

		/** Vector containing the modified mesh data */
		std::vector<trisoup::CallTriMeshData::Mesh> meshVector;
	};

} /* end namespace sombreros */
} /* end namespace megamol */



#endif