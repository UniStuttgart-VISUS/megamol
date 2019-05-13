/*
 * Constest2019DataLoader.h
 *
 * Copyright (C) 2019 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CONTEST2019DATALOADER_H_INCLUDED
#define MEGAMOLCORE_CONTEST2019DATALOADER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "astro/AstroDataCall.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/AnimDataModule.h"
#include "vislib/math/Cuboid.h"

namespace megamol {
namespace astro {

class Contest2019DataLoader : public core::view::AnimDataModule {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "Contest2019DataLoader"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "Data source module for the data of the SciVis Contest 2019."; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    /** Ctor. */
    Contest2019DataLoader(void);

    /** Dtor. */
    virtual ~Contest2019DataLoader(void);

protected:
    virtual core::view::AnimDataModule::Frame* constructFrame(void) const;

    virtual bool create(void);

    virtual void loadFrame(core::view::AnimDataModule::Frame* frame, unsigned int idx);

    virtual void release(void);

    class Frame : public core::view::AnimDataModule::Frame {
    public:
        Frame(core::view::AnimDataModule& owner);

        virtual ~Frame(void);

        inline void Clear(void) {
            this->positions->clear();
            this->velocities->clear();
            this->temperatures->clear();
            this->masses->clear();
            this->internalEnergies->clear();
            this->smoothingLengths->clear();
            this->molecularWeights->clear();
            this->densities->clear();
            this->gravitationalPotentials->clear();
            this->isBaryonFlags->clear();
            this->isStarFlags->clear();
            this->isWindFlags->clear();
            this->isStarFormingGasFlags->clear();
            this->isAGNFlags->clear();
            this->particleIDs->clear();
            // TODO shrink to fit?
        }

        /**
         * Loads a frame from a given file into this object
         *
         * @param filepath The path to the file to load from. As all frames are in seperate files, no open stream is
         * necessary.
         * @param frameIdx The zero-based index of the loaded frame.
         *
         * @return True on success, false otherwise.
         */
        bool LoadFrame(std::string filepath, unsigned int frameIdx);

        /**
         * Sets the data pointers of a given call to the internally stored values
         *
         * @param call The call that will be filled with the new data
         */
        void SetData(AstroDataCall& call);

    private:
        /** Pointer to the position array */
        vec3ArrayPtr positions = nullptr;

        /** Pointer to the velocity array */
        vec3ArrayPtr velocities = nullptr;

        /** Pointer to the temperature array */
        floatArrayPtr temperatures = nullptr;

        /** Pointer to the mass array */
        floatArrayPtr masses = nullptr;

        /** Pointer to the interal energy array */
        floatArrayPtr internalEnergies = nullptr;

        /** Pointer to the smoothing length array */
        floatArrayPtr smoothingLengths = nullptr;

        /** Pointer to the molecular weight array */
        floatArrayPtr molecularWeights = nullptr;

        /** Pointer to the density array */
        floatArrayPtr densities = nullptr;

        /** Pointer to the gravitational potential array */
        floatArrayPtr gravitationalPotentials = nullptr;

        /** Pointer to the baryon flag array */
        boolArrayPtr isBaryonFlags = nullptr;

        /** Pointer to the star flag array */
        boolArrayPtr isStarFlags = nullptr;

        /** Pointer to the wind flag array */
        boolArrayPtr isWindFlags = nullptr;

        /** Pointer to the star forming gas flag array */
        boolArrayPtr isStarFormingGasFlags = nullptr;

        /** Pointer to the AGN flag array */
        boolArrayPtr isAGNFlags = nullptr;

        /** Pointer to the particle ID array */
        idArrayPtr particleIDs = nullptr;
    };

    class Unlocker : public AstroDataCall::Unlocker {
    public:
        Unlocker(Frame& frame) : AstroDataCall::Unlocker(), frame(&frame) {
            // intentionally empty
        }

        virtual ~Unlocker(void) {
            this->Unlock();
            ASSERT(this->frame == nullptr);
        }

        virtual void Unlock(void) {
            if (this->frame != nullptr) {
                this->frame->Unlock();
                this->frame = nullptr; // DO NOT DELETE!
            }
        }

    private:
        Frame* frame;
    };

    bool getDataCallback(core::Call& caller);

    bool getExtentCallback(core::Call& caller);

    bool filenameChangedCallback(core::param::ParamSlot& slot);

    core::param::ParamSlot firstFilename;

    core::param::ParamSlot filesToLoad;

    core::CalleeSlot getDataSlot;

    vislib::math::Cuboid<float> boundingBox;

    vislib::math::Cuboid<float> clipBox;

    size_t data_hash;

    std::vector<std::string> filenames;
};

} // namespace astro
} // namespace megamol

#endif /* MEGAMOLCORE_CONTEST2019DATALOADER_H_INCLUDED */
