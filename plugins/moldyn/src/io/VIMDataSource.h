/*
 * VIMDataSource.h
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_VIMDATASOURCE_H_INCLUDED
#define MEGAMOLCORE_VIMDATASOURCE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "geometry_calls/MultiParticleDataCall.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd/data/AnimDataModule.h"
#include "vislib/RawStorage.h"
#include "vislib/sys/FastFile.h"
#include "vislib/sys/File.h"
#include "vislib/types.h"
#include <vector>


namespace megamol {
namespace moldyn {
namespace io {

/**
 * Renderer for rendering the vis logo into the unit cube.
 */
class VIMDataSource : public core::view::AnimDataModule {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "VIMDataSource";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Data source module for VIM files.";
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
    VIMDataSource();

    /** Dtor. */
    ~VIMDataSource() override;

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
    /** Nested class of simple vim types */
    class SimpleType {
    public:
        /** Ctor */
        SimpleType() : id(0), rad(0.0f) {
            // intentionally empty
        }

        /** Dtor */
        ~SimpleType() {
            // intentionally empty
        }

        /**
         * Gets the blue colour component of the type
         *
         * @return The blue colour component of the type
         */
        inline unsigned char Blue() const {
            return this->col[2];
        }

        /**
         * Gets the RGB colour of the type
         *
         * @return The RGB colour of the type
         */
        inline const unsigned char* Colour() const {
            return this->col;
        }

        /**
         * Gets the green colour component of the type
         *
         * @return The green colour component of the type
         */
        inline unsigned char Green() const {
            return this->col[1];
        }

        /**
         * Gets the id of the type
         *
         * @return The id of the type
         */
        inline unsigned int ID() const {
            return this->id;
        }

        /**
         * Gets the radius of the type
         *
         * @return The radius of the type
         */
        inline float Radius() const {
            return this->rad;
        }

        /**
         * Gets the red colour component of the type.
         *
         * @return The red colour component of the type.
         */
        inline unsigned char Red() const {
            return this->col[0];
        }

        /**
         * Sets the colour for the type.
         *
         * @param col The new RGB colour for the type.
         */
        inline void SetColour(const unsigned char* col) {
            if (col == NULL) {
                this->col[0] = this->col[1] = this->col[2] = 0;
            } else {
                this->col[0] = col[0];
                this->col[1] = col[1];
                this->col[2] = col[2];
            }
        }

        /**
         * Sets the colour for the type.
         *
         * @param col The new RGB colour for the type.
         */
        inline void SetColour(UINT32 col) {
            unsigned char* c = reinterpret_cast<unsigned char*>(&col);
            this->col[0] = c[0]; // TODO: Unsure with order!
            this->col[1] = c[1];
            this->col[2] = c[2];
        }

        /**
         * Sets the colour for the type.
         *
         * @param r The new red colour component
         * @param g The new green colour component
         * @param b The new blue colour component
         */
        inline void SetColour(unsigned char r, unsigned char g, unsigned char b) {
            this->col[0] = r;
            this->col[1] = g;
            this->col[2] = b;
        }

        /**
         * Sets the id for the type
         *
         * @param id The new if for the type.
         */
        inline void SetID(unsigned int id) {
            this->id = id;
        }

        /**
         * Sets the radius for the type.
         *
         * @param rad The new radius for the type.
         */
        inline void SetRadius(float rad) {
            this->rad = rad;
        }

        /**
         * Assignment operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return Reference to this.
         */
        SimpleType& operator=(const SimpleType& rhs) {
            this->col[0] = rhs.col[0];
            this->col[1] = rhs.col[1];
            this->col[2] = rhs.col[2];
            this->id = rhs.id;
            this->rad = rhs.rad;
            return *this;
        }

        /**
         * Test for equality
         *
         * @param rhs The right hand side operand.
         *
         * @return 'true' if 'this' equals 'rhs', 'false' otherwise.
         */
        bool operator==(const SimpleType& rhs) {
            return this->id == rhs.id; // id is currently enought to compare
        }

    private:
        /** The colour of the type */
        unsigned char col[3];

        /** The id of the type */
        unsigned int id;

        /** The radius of spheres of this type */
        float rad;
    };

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
         * Clears the internal data buffers
         */
        void Clear();

        /**
         * Loads a frame from 'file' to this object.
         *
         * @param file The data file.
         * @param idx The index number of the frame.
         * @param types The types array of the data.
         * @param scaling The global scaling of the bounding box.
         *
         * @return 'true' on success, 'false' on failure.
         */
        bool LoadFrame(vislib::sys::File* file, unsigned int idx, SimpleType* types, float scaling);

        /**
         * Sets the number of types of the data set.
         *
         * @param cnt The number of types of the data set.
         */
        void SetTypeCount(unsigned int cnt);

        /**
         * Gets the number of particles for the requested type.
         *
         * @param type The requested type index.
         *
         * @return The number of particles.
         */
        inline unsigned int PartCnt(unsigned int type) const {
            ASSERT(type < this->typeCnt);
            return this->partCnt[type];
        }

        /**
         * Gets the array of particle positions for the requested type.
         *
         * @param type The requested type index.
         *
         * @return The array of particle positions.
         */
        const float* PartPoss(unsigned int type) const;

        /**
         * Gets the array of particle quaternions for the requested type.
         *
         * @param type The requested type index.
         *
         * @return The array of particle quaternions.
         */
        const float* PartQuats(unsigned int type) const;

        /**
         * Returns radii data (three radii per particle). Data is generated if necessary.
         *
         * @param type The requested type index.
         *
         * @return The array of particle radii.
         */
        const float* PartRadii(unsigned int type, SimpleType& t) const;

        /**
         * Answers the size of the loaded data in bytes.
         *
         * @return The size of the loaded data in bytes.
         */
        SIZE_T SizeOf() const;

        /**
         * Replaces the data of this object with the interpolated data
         * based on the two frames 'a', 'b', and the interpolation
         * parameter 'alpha' [0, 1].
         *
         * @param alpha The interpolation parameter.
         * @param a The first interpolation value, used if 'alpha' is zero.
         * @param b The second interpolation value, used if 'alpha' is one.
         *
         * @return The frame to be used after the interpolation.
         */
        const Frame* MakeInterpolationFrame(float alpha, const Frame& a, const Frame& b);

    private:
        /**
         * Parses a particle line of the vim file.
         *
         * @param line The line to parse.
         * @param outType Receives the type of the parse particle.
         * @param outX Receives the x component of the position of the
         *             parsed particle.
         * @param outY Receives the y component of the position of the
         *             parsed particle.
         * @param outZ Receives the z component of the position of the
         *             parsed particle.
         * @param outQX Receives the x component of the orientation
         *              quaternion of the parsed particle.
         * @param outQY Receives the y component of the orientation
         *              quaternion of the parsed particle.
         * @param outQZ Receives the z component of the orientation
         *              quaternion of the parsed particle.
         * @param outQW Receives the w component of the orientation
         *              quaternion of the parsed particle.
         */
        void parseParticleLine(vislib::StringA& line, int& outType, float& outX, float& outY, float& outZ, float& outQX,
            float& outQY, float& outQZ, float& outQW);

        /** type count */
        unsigned int typeCnt;

        /** particle counts per type */
        unsigned int* partCnt;

        /** position data per type */
        vislib::RawStorage* pos;

        /** quaternion data per type */
        vislib::RawStorage* quat;

        mutable std::vector<vislib::RawStorage> radii;
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

    /** Calculates the bounding box from all frames. */
    void calcBoundingBox();

    /**
     * Callback receiving the update of the file name parameter.
     *
     * @param slot The updated ParamSlot.
     *
     * @return Always 'true' to reset the dirty flag.
     */
    bool filenameChanged(core::param::ParamSlot& slot);

    /**
     * Parses a type description line of the vim file.
     *
     * @param line The line to parse
     * @param outType Receives the type of the parsed element.
     *
     * @return The SimpleType object representing this type line
     *         or 'NULL' in case of an error.
     */
    SimpleType* parseTypeLine(vislib::StringA& line, int& outType);

    /**
     * Reads the file header containing the particle descriptions.
     *
     * @param filename The file that is currently loading
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool readHeader(const vislib::TString& filename);

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

    /** The file name */
    core::param::ParamSlot filename;

    /** The slot for requesting data */
    core::CalleeSlot getData;

    /** The opened data file */
    vislib::sys::File* file;

    /** The number of types */
    unsigned int typeCnt;

    /** The types */
    SimpleType* types;

    /** The frame index table */
    vislib::sys::File::FileSize* frameIdx;

    /** Scaling from the unit box [0, 1] to the data sets bounding box (first frame) */
    float boxScaling;

    /** The data file hash */
    SIZE_T datahash;
};

} /* end namespace io */
} /* end namespace moldyn */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_VIMDATASOURCE_H_INCLUDED */
