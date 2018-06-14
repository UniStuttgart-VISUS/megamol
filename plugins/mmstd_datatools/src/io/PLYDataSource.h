/*
 * PLYDataSource.h
 *
 * Copyright (C) 2018 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_DATATOOLS_IO_PLYDATASOURCE_H_INCLUDED
#define MEGAMOL_DATATOOLS_IO_PLYDATASOURCE_H_INCLUDED
#pragma once

#include "mmcore/view/AnimDataModule.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/CalleeSlot.h"
#include "mmstd_datatools/GraphDataCall.h"
#include "vislib/sys/File.h"
#include "tinyply.h"
#include <cstdint>
#include <vector>


namespace megamol {
namespace stdplugin {
namespace datatools {
namespace io {

    /**
     * Data source module for .PLY files.
     */
    class PLYDataSource : public core::Module {
    public:

        /**
         * Returns the name of the class.
         *
         * @return The name of the class.
         */
        static const char *ClassName(void) { 
            return "PLYDataSource"; 
        }

        /**
         * Returns the description of the module.
         *
         * @return The description of the module.
         */
        static const char *Description(void) { 
            return "Data source module for .PLY files."; 
        }

        /**
         * Returns whether the module is available.
         *
         * @return True, if the module is available. False otherwise.
         */
        static bool IsAvailable(void) { 
            return true; 
        }

        /**
         * Constructor.
         */
        PLYDataSource(void);

        /**
         * Destructor
         */
        virtual ~PLYDataSource(void);


    protected:

        /**
         * Creates the module.
         *
         * @return True on success. False otherwise.
         */
        virtual bool create(void);

        /**
         * Destroys the module.
         */
        virtual void release(void);

        ///**
        // * Helper class to unlock frame data when 'CallSimpleSphereData' is
        // * used.
        // */
        //class Unlocker : public GraphDataCall::Unlocker {
        //public:
        //    Unlocker(Frame& frame) : GraphDataCall::Unlocker(), frame(&frame) {
        //        // intentionally empty
        //    }
        //    virtual ~Unlocker(void) {
        //        this->Unlock();
        //        ASSERT(this->frame == nullptr);
        //    }
        //    virtual void Unlock(void) {
        //        if (this->frame != nullptr) {
        //            this->frame->Unlock();
        //            this->frame = nullptr;
        //        }
        //    }
        //private:
        //    Frame *frame;
        //};

        /**
         * Callback called when the filename is changed.
         * 
         * @param slot The slot triggering the this callback.
         * @return True on success. False otherwise.
         */
        bool filenameChanged(core::param::ParamSlot& slot);

        /**
         * Reimplementation of the getData callback.
         * 
         * @param caller The triggering call.
         * @return True on success. False otherwise.
         */
        bool getMeshDataCallback(core::Call& caller);

        /**
         * Reimplementation of the getExtent callback.
         *
         * @param The triggering call.
         * @return True on success. False otherwise.
         */
        bool getMeshExtentCallback(core::Call& caller);

        /**
         * Reimplementation of the getData callback.
         * 
         * @param caller The triggering call.
         * @return True on success. False otherwise.
         */
        bool getSphereDataCallback(core::Call& caller);

        /**
         * Reimplementation of the getExtent callback.
         *
         * @param The triggering call.
         * @return True on success. False otherwise.
         */
        bool getSphereExtentCallback(core::Call& caller);

        /**
         * clears all fields
         */
        void clearAllFields(void);

        /** The slot for the filename */
        core::param::ParamSlot filename;
        
        /** The slot to put out the mesh data */
        core::CalleeSlot getDataMesh;

        /** The slot to put out the vertex data */
        core::CalleeSlot getDataSpheres;

        /** Pointer to the vislib file */
        vislib::sys::File *file;

        /** The current data hash */
        size_t data_hash;

        /** Struct for the different possible position types */
        struct pos_type {
            double * pos_double = nullptr;
            float * pos_float = nullptr;
        } posPointers;

        /** Struct for the different possible color types */
        struct col_type {
            unsigned char * col_uchar = nullptr;
            float * col_float = nullptr;
            double * col_double = nullptr;
        } colorPointers;

        /** Struct for the different possible normal types */
        struct normal_type {
            double * norm_double = nullptr;
            float * norm_float = nullptr;
        } normalPointers;

        /** Struct for the different possible face types */
        struct face_type {
            uint8_t * face_uchar = nullptr;
            uint16_t * face_u16 = nullptr;
            uint32_t * face_u32 = nullptr;
        } facePointers;

        /** Pointer to the face data */
        face_type faces;

        /** Pointer to the vertex color data */
        col_type vertColors;

        /** Pointer to the vertex normal data */
        normal_type vertNormals;

        /** Pointer to the vertex position data */
        pos_type vertPositions;
    };

} /* end namespace io */
} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* MEGAMOL_DATATOOLS_IO_PLYDATASOURCE_H_INCLUDED */
