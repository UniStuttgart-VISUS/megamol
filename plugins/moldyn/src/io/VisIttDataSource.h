/*
 * VisIttDataSource.h
 *
 * Copyright (C) 2009-2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_VISITTDATASOURCE_H_INCLUDED
#define MEGAMOLCORE_VISITTDATASOURCE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "geometry_calls/MultiParticleDataCall.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd/data/AnimDataModule.h"
#include "vislib/Array.h"
#include "vislib/Pair.h"
#include "vislib/RawStorage.h"
#include "vislib/math/Cuboid.h"
#include "vislib/math/mathfunctions.h"
#include "vislib/sys/File.h"
#include <cassert>
#include <map>
#include <vector>


namespace megamol {
namespace moldyn {
namespace io {


/**
 * Renderer for rendering the vis logo into the unit cube.
 */
class VisIttDataSource : public core::view::AnimDataModule {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "VisIttDataSource";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Data source module for VisItt files.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    /** Ctor. */
    VisIttDataSource();

    /** Dtor. */
    ~VisIttDataSource() override;

protected:
    /**
     * Creates a frame to be used in the frame cache. This method will be
     * called from within 'initFrameCache'.
     *
     * @return The newly created frame object.
     */
    core::view::AnimDataModule::Frame* constructFrame() const override;

    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

    /**
     * Loads one frame of the data set into the given 'frame' object. This
     * method may be invoked from another thread. You must take
     * precausions in case you need synchronised access to shared
     * ressources.
     *
     * @param frame The frame to be loaded.
     * @param idx The index of the frame to be loaded.
     */
    void loadFrame(core::view::AnimDataModule::Frame* frame, unsigned int idx) override;

    /**
     * Implementation of 'Release'.
     */
    void release() override;

private:
    /** Nested class of frame data */
    class Frame : public core::view::AnimDataModule::Frame {
    public:
        /**
         * Ctor.
         *
         * @param owner The owning AnimDataModule
         */
        Frame(core::view::AnimDataModule& owner);

        /** Dtor. */
        ~Frame() override;

        /**
         * Clears the particle data
         */
        inline void Clear() {
            this->frame = 0;
            this->dat.clear();
        }

        /**
         * Sets the frame number
         *
         * @param fn The frame number
         */
        inline void SetFrameNumber(unsigned int fn) {
            this->frame = fn;
        }

        /**
         * Gets the frame number
         *
         * @return frame number
         */
        inline unsigned int getFrameNumber() {
            return this->frame;
        }

        /**
         * Gets the particle count of typeId
         *
         * @param typeId The id of this type
         *
         * @return The number of particles
         */
        inline unsigned int ParticleCount(unsigned int typeId) const {
            assert((dat.at(typeId).size() % 3) == 0);
            return static_cast<unsigned int>(dat.at(typeId).size() / 3);
        }

        /**
         * Gets the particle data of typeId
         *
         * @param typeId The id of this type
         *
         * @return The data structure holding the position data
         */
        inline const float* ParticleData(unsigned int typeId) const {
            return dat.at(typeId).data();
        }

        /**
         * Access the particle data of typeId.
         * Also creates the required data structure if not present
         *
         * @param typeId The id of this type
         *
         * @return The data structure holding the position data
         */
        inline std::vector<float>& AccessParticleData(unsigned int typeId) {
            return dat[typeId];
        }

        /**
         * Answer all stored particle types
         *
         * @return All stored particle types
         */
        inline std::vector<unsigned int> ParticleTypes() const {
            std::vector<unsigned int> keys;
            for (const std::pair<unsigned int, std::vector<float>>& p : dat) {
                keys.push_back(p.first);
            }
            return keys;
        }

        /**
         * Answer the approximate in memory frame data size
         *
         * @return The approximate size of the frame data
         */
        inline uint64_t GetFrameSize() const {
            uint64_t fs = 0;
            for (const std::pair<unsigned int, std::vector<float>>& p : dat) {
                fs += p.second.size() * sizeof(float);
            }
            return fs;
        }

    private:
        /** The size of the memory really used */
        SIZE_T size;

        /** The xyz particle positions */
        std::map<unsigned int, std::vector<float>> dat;
    };

    /**
     * Helper class to unlock frame data when 'CallSimpleSphereData' is
     * used.
     */
    class Unlocker : public geocalls::MultiParticleDataCall::Unlocker {
    public:
        /**
         * Ctor.
         *
         * @param frame The frame to unlock
         */
        Unlocker(Frame& frame) : geocalls::MultiParticleDataCall::Unlocker(), frame(&frame) {
            // intentionally empty
        }

        /** Dtor. */
        ~Unlocker() override {
            this->Unlock();
            ASSERT(this->frame == NULL);
        }

        /** Unlocks the data */
        void Unlock() override {
            if (this->frame != NULL) {
                this->frame->Unlock();
                this->frame = NULL; // DO NOT DELETE!
            }
        }

    private:
        /** The frame to unlock */
        Frame* frame;
    };

    /** Builds up the frame index table. */
    void buildFrameTable();

    ///** Calculates the bounding box from all frames. */
    //void calcBoundingBox(void);

    /**
     * Callback receiving the update of the file name parameter.
     *
     * @param slot The updated ParamSlot.
     *
     * @return Always 'true' to reset the dirty flag.
     */
    bool filenameChanged(core::param::ParamSlot& slot);

    /**
     * The filter settings changed, so we need to reload all data
     *
     * @param slot The slot
     *
     * @return 'true'
     */
    bool filterChanged(core::param::ParamSlot& slot);

    /**
     * Parses the file header containing the particle descriptions
     *
     * @param header The file header line
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool parseHeader(const vislib::StringA& header);

    /**
     * Gets the data from the source.
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool getDataCallback(core::Call& caller);

    /**
     * Gets the data from the source.
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool getExtentCallback(core::Call& caller);

    /** Finds the data column used for filtering */
    void findFilterColumn();

    /** Finds the data column used for the particle index */
    void findIdColumn();

    /** Finds the data column used for the color number */
    void findTypeColumn();

    /** The file name */
    core::param::ParamSlot filename;

    /** The radius for the particles */
    core::param::ParamSlot radius;

    /** The filter to be applied */
    core::param::ParamSlot filter;

    /** The filter column to be applied */
    core::param::ParamSlot filterColumn;

    /** The filter value to be applied */
    core::param::ParamSlot filterValue;

    /** The slot for requesting data */
    core::CalleeSlot getData;

    /** The opened data file */
    vislib::sys::File* file;

    /** The file data hash number */
    SIZE_T dataHash;

    /** The frame seek table */
    vislib::Array<vislib::sys::File::FileSize> frameTable;

    /** All column widths and labels */
    vislib::Array<vislib::Pair<vislib::StringA, unsigned int>> header;

    /** The sorted index of the usable columns */
    vislib::Array<unsigned int> headerIdx;

    /** The column index used for filtering */
    unsigned int filterIndex;

    /** The bounding box */
    vislib::math::Cuboid<float> bbox;

    /** Flag to sort particles based on their IDs */
    core::param::ParamSlot sortPartIdSlots;

    /** The id column index */
    unsigned int idIndex;

    /** Flag to sort particles based on their IDs */
    core::param::ParamSlot splitTypesSlots;

    /** Flag to sort particles based on their IDs */
    core::param::ParamSlot splitTypesNameSlots;

    /** The color number column index */
    unsigned int typeIndex;
};

} /* end namespace io */
} /* end namespace moldyn */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_VISITTDATASOURCE_H_INCLUDED */
