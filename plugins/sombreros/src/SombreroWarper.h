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
#include "protein_calls/BindingSiteCall.h"
#include "TunnelResidueDataCall.h"

#include <set>

namespace megamol {
namespace sombreros {

	typedef unsigned int uint;

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

		/**
		 * Checks for dirty parameters and sets the dirty flag, if needed
		 */
		void checkParameters(void);

		/**
		 * Copies the mesh data into local buffers to make modification possible.
		 *
		 * @param ctmd The incoming call with the source data.
		 * @return True on success, false otherwise.
		 */
		bool copyMeshData(megamol::trisoup::CallTriMeshData& ctmd);

		/**
		 * Searches for the outer vertices of the sombrero brim.
		 *
		 * @return True on success, false otherwise.
		 */
		bool findSombreroBorder(void);

		/**
		 * Warps the mesh to resemble a sombrero.
		 *
		 * @bsCall Call containing the binding site data.
		 * @tunnelCall Call containing the tunnel data.
		 * @return True on success, false otherwise.
		 */
		bool warpMesh(protein_calls::BindingSiteCall& bsCall, TunnelResidueDataCall& tunnelCall);

		/** The lastly received data hash */
		SIZE_T lastDataHash;

		/** The offset to the lastly received hash */
		SIZE_T hashOffset;

		/** Slot for the mesh input. */
		core::CallerSlot meshInSlot;

		/** Slot for the binding site input. */
		core::CallerSlot bindingSiteInSlot;

		/** Slot for the tunnel input */
		core::CallerSlot tunnelInSlot;

		/** Slot for the ouput of the cut mesh */
		core::CalleeSlot warpedMeshOutSlot;

		/** Parameter for the minimal brim level */
		core::param::ParamSlot minBrimLevelParam;

		/** Parameter fo the maximum brim level */
		core::param::ParamSlot maxBrimLevelParam;

		/** Vector containing the modified mesh data */
		std::vector<trisoup::CallTriMeshData::Mesh> meshVector;

		/** The vertex positions of the mesh */
		std::vector<std::vector<float>> vertices;

		/** The vertex normals of the mesh */
		std::vector<std::vector<float>> normals;
		
		/** The vertex colors of the mesh */
		std::vector<std::vector<unsigned char>> colors;

		/** The atom indices per vertex for the mesh */
		std::vector<std::vector<uint>> atomIndexAttachment;

		/** The vertex levels of the mesh */
		std::vector<std::vector<uint>> vertexLevelAttachment;
		
		/** The faces of the mesh */
		std::vector<std::vector<uint>> faces;

		/** The face edges in forward order */
		std::vector<std::vector<std::pair<uint, uint>>> edgesForward;

		/** The face edges in reverse order */
		std::vector<std::vector<std::pair<uint, uint>>> edgesReverse;

		/** Set containing the indices of the border vertices */
		std::vector<std::set<uint>> borderVertices;

		/** Flags for each vertex if they belong to the brim */
		std::vector<std::vector<bool>> brimFlags;

		/** Flag set when a parameter is dirty */
		bool dirtyFlag;
	};

} /* end namespace sombreros */
} /* end namespace megamol */



#endif