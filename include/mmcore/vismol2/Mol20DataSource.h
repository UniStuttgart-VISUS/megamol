/*
 * Mol20DataSource.h
 *
 * Copyright (C) 2008 - 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_MOL20DATASOURCE_H_INCLUDED
#define MEGAMOLCORE_MOL20DATASOURCE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/view/AnimDataModule.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/vismol2/Mol2Data.h"
#include "vislib/sys/File.h"
#include "vislib/types.h"
#include "vislib/math/Cuboid.h"


namespace megamol {
namespace core {
namespace vismol2 {


    /* HAZARD */
    #define NUM_MOLECULE_TYPES 200

    //extern MOL_COUNTER num_molecules;

    //extern molecule_t molecules[NUM_MOLECULE_TYPES];
    //extern molecule_upload_t upmols[NUM_MOLECULE_TYPES];

    //extern float molsize_offset;
    //extern float molsize_factor;


    /**
     * Loader module for Mol2.0 data files.
     */
    class Mol20DataSource : public view::AnimDataModule {
    public:

        /**
         * Nested class storing one Mol2 frame.
         */
        class Frame : public view::AnimDataModule::Frame {
        public:

            friend class Mol20DataSource;

            /**
             * Ctor.
             *
             * @param owner The owning AnimDataModule
             * @param pfc The point format count
             * @param pf The point format
             * @param withQ1 The flag if quaternion data is present
             */
            Frame(view::AnimDataModule& owner, const int& pfc,
                const int *const& pf, const int& withQ1);

            /** Dtor. */
            virtual ~Frame(void);

            /** The frame data */
            pointcloud_t data;

            /** The bounding box */
            vislib::math::Cuboid<float> bbox;

            /** clears the frame */
            void clear(void);

            /**
             * clears a cluster struct including all children
             *
             * @param c The cluster to clear.
             */
            void clearCluster(cluster_t *c);

            /**
             * Loads a cluster data from the file.
             *
             * @param c The cluster to load
             * @param p
             * @param file The file to load from.
             *
             * @return Number of byte read.
             */
            vislib::sys::File::FileSize LoadCluster(cluster_t *c, point_t *p, vislib::sys::File& file);

            /**
             * Loads point information from the file.
             *
             * @param p
             * @param file The file to load from.
             *
             * @return Number of byte read.
             */
            vislib::sys::File::FileSize LoadPoint(point_t *p, vislib::sys::File& file);

            /**
             * Answers the size of this frame in memory.
             *
             * @return The size of this frame in memory.
             */
            unsigned int InMemSize(void) const;

            /** Calculates the frames bounding box */
            void CalcBBox(void);

        private:

            /**
             * Answers the size of this frame in memory.
             *
             * @return The size of this frame in memory.
             */
            unsigned int InMemSize(const cluster_t& c) const;

            const int& pointFormatSize;
            const int *const& pointFormat;
            const int& hasQ1;

        };

        /**
         * Helper class to store flux data
         */
        class FluxStore {
        public:

            /** Ctor */
            FluxStore(void);

            /** Dtor */
            ~FluxStore(void);

            /**
             * Sets the number of time frames in the data set.
             *
             * @param cnt The number of time frames.
             */
            void SetFrameCnt(unsigned int cnt);

            /** Clears the flux store */
            void Clear(void);

            /**
             * Adds one flux object
             *
             * @param sx The starting x coordinate
             * @param sy The starting y coordinate
             * @param sz The starting z coordinate
             * @param st The starting time
             * @param tx The targetted x coordinate
             * @param ty The targetted y coordinate
             * @param tz The targetted z coordinate
             * @param tt The targetted time
             * @param cnt The number of molecules
             * @param sp The starting size percentage
             * @param tp The targetted size percentage
             */
            void AddFlux(float sx, float sy, float sz, int st, float tx, float ty, float tz, int tt, int cnt, float sp, float tp);
        };

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "Mol20DataSource";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Data source module for Mol2.0 files.";
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
        Mol20DataSource(void);

        /** Dtor. */
        virtual ~Mol20DataSource(void);

    protected:

        /**
         * Creates a frame to be used in the frame cache. This method will be
         * called from within 'initFrameCache'.
         *
         * @return The newly created frame object.
         */
        virtual view::AnimDataModule::Frame* constructFrame(void) const;

        /**
         * Implementation of 'Create'.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        virtual bool create(void);

        /**
         * Loads one frame of the data set into the given 'frame' object. This
         * method may be invoked from another thread. You must take 
         * precausions in case you need synchronised access to shared 
         * ressources.
         *
         * @param frame The frame to be loaded.
         * @param idx The index of the frame to be loaded.
         */
        virtual void loadFrame(view::AnimDataModule::Frame *frame,
            unsigned int idx);

        /**
         * Implementation of 'Release'.
         */
        virtual void release(void);

    private:

        /**
         * Callback receiving the update of the file name parameter.
         *
         * @param slot The updated ParamSlot.
         *
         * @return Always 'true' to reset the dirty flag.
         */
        bool filenameChanged(param::ParamSlot& slot);

        /**
         * Gets the data from the source.
         *
         * @param caller The calling call.
         *
         * @return 'true' on success, 'false' on failure.
         */
        bool getDataCallback(Call& caller);

        /**
         * Gets the data from the source.
         *
         * @param caller The calling call.
         *
         * @return 'true' on success, 'false' on failure.
         */
        bool getExtentCallback(Call& caller);

        /** The file name */
        param::ParamSlot filename;

        /** The opened data file */
        vislib::sys::File *file;

        /** The bounding box */
        vislib::math::Cuboid<float> bbox;

        /** The flux store */
        FluxStore flux;

        /** The slot for requesting data */
        CalleeSlot getData;

        /** ported members */
        MOL_COUNTER num_molecules;
        molecule_t molecules[NUM_MOLECULE_TYPES];
        molecule_upload_t upmols[NUM_MOLECULE_TYPES];
        float molsize_offset;
        float molsize_factor;
        int pointFormatSize;
        int *pointFormat;
        vislib::sys::File::FileSize *frameTable;
        int hasQ1;

    };


} /* end namespace vismol2 */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_MOL20DATASOURCE_H_INCLUDED */
