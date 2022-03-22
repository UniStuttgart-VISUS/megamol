/*
 * CrystalStructureDataSource.h
 *
 * Copyright (C) 2012 by University of Stuttgart (VISUS).
 * All rights reserved.
 *
 * $Id$
 */

#ifndef MMPROTEINPLUGIN_CRYSTALSTRUCTUREDATASOURCE_H_INCLUDED
#define MMPROTEINPLUGIN_CRYSTALSTRUCTUREDATASOURCE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/AnimDataModule.h"
#include "protein_calls/CrystalStructureDataCall.h"
#include "vislib/Array.h"
#include "vislib/math/Vector.h"
#include <fstream>

namespace megamol {
namespace protein {

/**
 * TODO
 */
class CrystalStructureDataSource : public core::view::AnimDataModule {
public:
    /** Ctor */
    CrystalStructureDataSource(void);

    /** Dtor */
    virtual ~CrystalStructureDataSource(void);

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "CrystalStructureDataSource";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Offers data read from trajectory files.";
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
    /// Source for (approximated) dipoles
    enum DipoleSrc { DIPOLE_DISPL, DIPOLE_DISPLTI, DIPOLE_CELL, DIPOLE_NOBA, DIPOLE_BATI, CHKPT_SOURCE, VECFIELD_PROC };

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

    /**
     * Call callback to get the extent of the data
     *
     * @param call The calling call
     *
     * @return True on success
     */
    bool getExtent(core::Call& call);

    /**
     * Implementation of 'Release'.
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
    virtual void loadFrame(Frame* frame, unsigned int idx);

private:
    /**
     * Nested class to store frame data.
     */
    class Frame : public core::view::AnimDataModule::Frame {
    public:
        /** Ctor */
        Frame(megamol::core::view::AnimDataModule& owner);

        /** Dtor */
        virtual ~Frame(void);

        /**
         * Answer the name of this class.
         *
         * @return The name of this class.
         */
        static const char* ClassName(void) {
            return "CrystalStructureDataSource::Frame";
        }

        /**
         * Returns frame data.
         *
         * @return Pointer to the frame data.
         */
        const float* GetAtomPos() const {
            return this->atomPos;
        }

        /**
         * Returns the dipole array.
         *
         * @return The dipole array
         */
        const float* GetDipole() const {
            return this->dipole;
        }

        /**
         * Returns the array with the dipole positions.
         *
         * @return The dipole positions
         */
        const float* GetDipolePos() const {
            return this->dipolePos;
        }

        /**
         * Answers the frames index.
         *
         * @return The frame index.
         */
        unsigned int GetFrameIdx() {
            return this->frame;
        }

        /**
         * Sets the frame index.
         *
         * @param idx The frame index.
         */
        void SetFrameIdx(unsigned int idx) {
            this->frame = idx;
        }

        /**
         * TODO
         */
        void SetAtomPosAtIdx(unsigned int idx, float x, float y, float z) {
            this->atomPos[idx * 3 + 0] = x;
            this->atomPos[idx * 3 + 1] = y;
            this->atomPos[idx * 3 + 2] = z;
        }

        /**
         * TODO
         */
        void SetDipolePosAtIdx(unsigned int idx, float x, float y, float z) {
            this->dipolePos[idx * 3 + 0] = x;
            this->dipolePos[idx * 3 + 1] = y;
            this->dipolePos[idx * 3 + 2] = z;
        }

        /**
         * TODO
         */
        void SetDipoleAtIdx(unsigned int idx, float x, float y, float z) {
            this->dipole[idx * 3 + 0] = x;
            this->dipole[idx * 3 + 1] = y;
            this->dipole[idx * 3 + 2] = z;
        }

        /**
         * TODO
         */
        void AllocBufs(unsigned int atomCnt, unsigned int dipoleCnt) {
            this->atomPos = new float[3 * atomCnt];
            this->dipole = new float[3 * dipoleCnt];
            this->dipolePos = new float[3 * dipoleCnt];
            this->atomCnt = atomCnt;
            this->dipoleCnt = dipoleCnt;
        }

    private:
        /// The atom positions
        float* atomPos;

        /// The dipole positions
        float* dipolePos;

        /// The dipoles
        float* dipole;

        /// The number of atoms in this frame
        unsigned int atomCnt;

        /// The number of dipoles in this frame
        unsigned int dipoleCnt;

        friend class CrystalStructureDataSource; // Because the data source needs to access private pointers
    };

    /**
     * Helper class to unlock frame data when 'CrystalStructureDataCall' is
     * used.
     */
    class Unlocker : public protein_calls::CrystalStructureDataCall::Unlocker {
    public:
        /**
         * Ctor.
         *
         * @param frame The frame to unlock
         */
        Unlocker(Frame& frame) : protein_calls::CrystalStructureDataCall::Unlocker(), frame(&frame) {
            // intentionally empty
        }

        /** Dtor. */
        virtual ~Unlocker(void) {
            this->Unlock();
            ASSERT(this->frame == NULL);
        }

        /** Unlocks the data */
        virtual void Unlock(void) {
            if (this->frame != NULL) {
                this->frame->Unlock();
                this->frame = NULL; // DO NOT DELETE!
            }
        }

    private:
        /** The frame to unlock */
        Frame* frame;
    };

    /**
     * Loads the atom and cell files and counts number of frames based on
     * the frame file.
     *
     * @return True, if all files could be loaded.
     */
    bool loadFiles();

    /**
     * Update parameters
     */
    void updateParams();

    /**
     * TODO
     */
    bool WriteFrameData(CrystalStructureDataSource::Frame* fr);

    /**
     * Helper funczion which converts std::string to var of type T.
     *
     * @param str The std::string.
     * @return The variable of type T.
     */
    template<class T>
    T convertStrTo(std::string str);


    // (Parameter) slots

    /// The data callee slot
    core::CalleeSlot dataOutSlot;

    /// The call for *.chkpt data
    core::CallerSlot dataChkptCallerSlot;

    /// The file name slot for frames
    core::param::ParamSlot fileFramesSlot;

    /// The file name slot for time independent atom data e.g. atom type
    core::param::ParamSlot fileAtomsSlot;

    /// The file name slot for cell information
    core::param::ParamSlot fileCellsSlot;

    /// The param slot for the frame cache size
    core::param::ParamSlot frameCacheSizeParam;
    int frameCacheSize;

    /// Parameter to change uni grid data base
    core::param::ParamSlot dSourceParam;
    DipoleSrc dSource;

    /// The param slot for the displacement offset
    core::param::ParamSlot displOffsParam;
    unsigned int displOffs;


    // Arrays

    /// The array for the connectivity information of all atoms
    vislib::Array<int> atomCon;

    // The vertex array for edges
    vislib::Array<int> atomEdges;

    /// The atom types
    vislib::Array<protein_calls::CrystalStructureDataCall::AtomType> atomType;

    /// The cell data
    int* cells;


    // Misc

    /// The bounding box
    vislib::math::Cuboid<float> bbox;

    /// The number of frames
    unsigned int frameCnt;

    /// The number of dipoles
    unsigned int dipoleCnt;

    /// The number of atoms
    unsigned int atomCnt;

    /// The number of cells
    unsigned int cellCnt;
};


} /* end namespace protein */
} /* end namespace megamol */

#endif // MMPROTEINPLUGIN_CRYSTALSTRUCTUREDATASOURCE_H_INCLUDED
