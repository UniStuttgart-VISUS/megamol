/*
 * MolecularAOShader.h
 *
 * Copyright (C) 2009 - 2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MMPROTEINCUDAPLUGIN_MOLECULARAOSHADER_H_INCLUDED
#define MMPROTEINCUDAPLUGIN_MOLECULARAOSHADER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "protein_calls/MolecularDataCall.h"

namespace megamol {
namespace protein_cuda {

    /**
     * Molecular Ambient Occlusion CPU based Shader.
     */
    class MolecularAOShader {
    public:
        /** Ctor. */
        MolecularAOShader(void);

        /** Dtor. */
        virtual ~MolecularAOShader(void);
		
        /**
         * Computes the AO Volume. The callee takes ownership of the returned 
		 * volume (delete[] it!)
         */
		float* createVolume(class megamol::protein_calls::MolecularDataCall& mol);
        /**
         * Computes the AO Volume. The callee takes ownership of the returned 
		 * volume (delete[] it!)
         */
		float* createVolumeDebug(class megamol::protein_calls::MolecularDataCall& mol);

		/**
         * Gets the AO volume's size in voxels.
         */
		int getVolumeSizeX() const;
		int getVolumeSizeY() const;
		int getVolumeSizeZ() const;

        /**
         * Sets the AO volume's size. in voxels.
         */
		void setVolumeSize(int volSizeX, int volSizeY, int volSizeZ);
		
        /**
         * Sets the generation factor (influence factor of a single sphere on a voxel).
         */
		void setGenerationFactor(float genFac);

	private:
        /** The size of the volume in numbers of voxels */
		int volSizeX;

		/** The size of the volume in numbers of voxels */
		int volSizeY;

		/** The size of the volume in numbers of voxels */
		int volSizeZ;

		/** The generation factor (influence factor of a single sphere on a voxel) */
		float genFac;
    };

} /* end namespace protein_cuda */
} /* end namespace megamol */

#endif /* MMPROTEINCUDAPLUGIN_MOLECULARAOSHADER_H_INCLUDED */
