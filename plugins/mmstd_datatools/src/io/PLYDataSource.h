/*
 * PLYDataSource.h
 *
 * Copyright (C) 2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_DATATOOLS_IO_PLYDATASOURCE_H_INCLUDED
#define MEGAMOL_DATATOOLS_IO_PLYDATASOURCE_H_INCLUDED
#pragma once

#include "mmcore/view/AnimDataModule.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/CalleeSlot.h"
#include "vislib/sys/File.h"
#include "vislib/sys/Log.h"
#include "tinyply.h"
#include <cstdint>
#include <vector>
#include <fstream>


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
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) { 
            return "PLYDataSource"; 
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) { 
            return "Data source module for .PLY files."; 
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) { 
            return true; 
        }

        /**
         * Constructor.
         */
        PLYDataSource(void);

        /**
         * Destructor.
         */
        virtual ~PLYDataSource(void);

    protected:

        /**
         * Implementation of 'Create'.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        virtual bool create(void);

        /**
         * Implementation of 'Release'.
         */
        virtual void release(void);

        /** 
         * Reads the data from an open instream.
         * 
         * @return True on success, false otherwise.
         */
        bool assertData(void);

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
         * Callback that is called when the filename is changed.
         * 
         * @param slot The slot containing the file path.
         * @return True on success, false otherwise.
         */
        bool filenameChanged(core::param::ParamSlot& slot);

        /**
         * Callback that is called when an enum is changed by the user.
         * 
         * @param slot The slot containing the changed enum.
         * @return True on success, false otherwise.
         */
        bool anyEnumChanged(core::param::ParamSlot& slot);

        /**
         * Callback setting the data of the read spheres.
         *
         * @param caller The calling call.
         * @return True on success, false otherwise.
         */
        bool getSphereDataCallback(core::Call& caller);

        /**
         * Callback setting the extent of the read sphere data.
         *
         * @param caller The calling call.
         * @return True on success, false otherwise.
         */
        bool getSphereExtentCallback(core::Call& caller);

        /**
         * Callback setting the data of the read mesh.
         *
         * @param caller The calling call.
         * @return True on success, false otherwise.
         */
        bool getMeshDataCallback(core::Call& caller);

        /**
         * Callback setting the extent of the read mesh data.
         *
         * @param caller The calling call.
         * @return True on success, false otherwise.
         */
        bool getMeshExtentCallback(core::Call& caller);

        /** Slot for the filepath of the .ply file */
        core::param::ParamSlot filename;

        /** Slot for the vertex element name */
        core::param::ParamSlot vertElemSlot;

        /** Slot for the face element name */
        core::param::ParamSlot faceElemSlot;

        /** Slot for the x property name */
        core::param::ParamSlot xPropSlot;

        /** Slot for the y property name */
        core::param::ParamSlot yPropSlot;

        /** Slot for the z property name */
        core::param::ParamSlot zPropSlot;

        /** Slot for the nx property name */
        core::param::ParamSlot nxPropSlot;

        /** Slot for the nx property name */
        core::param::ParamSlot nyPropSlot;

        /** Slot for the nz property name */
        core::param::ParamSlot nzPropSlot;

        /** Slot for the r property name */
        core::param::ParamSlot rPropSlot;

        /** Slot for the g property name */
        core::param::ParamSlot gPropSlot;

        /** Slot for the b property name */
        core::param::ParamSlot bPropSlot;

        /** Slot for the i property name */
        core::param::ParamSlot iPropSlot;

        /** Slot for the index property name */
        core::param::ParamSlot indexPropSlot;

        /** Guessed name of the position properties */
        std::vector<std::string> guessedPos;

        /** Guessed name of the normal properties */
        std::vector<std::string> guessedNormal;

        /** Guessed name of the color properties */
        std::vector<std::string> guessedColor;

        /** Guessed name of the index property */
        std::string guessedIndices;

        /** Guessed name of the vertex property */
        std::string guessedVertices;

        /** Guessed name of the face property */
        std::string guessedFaces;

        /// this also guards data availability and consistency with element/property selection!
        /** Pointer to the tinyply vertex data */
        std::shared_ptr<tinyply::PlyData> vertices;

        /** Pointer ot the tinyply normal data */
        std::shared_ptr<tinyply::PlyData> normals;

        /** Pointer to the tinyply color data */
        std::shared_ptr<tinyply::PlyData> colors;

        /** Pointer to the tinyply face data */
        std::shared_ptr<tinyply::PlyData> faces;

        /** Slot offering the sphere data. */
        core::CalleeSlot getData;

        /** Slot offering the mesh data. */
        core::CalleeSlot getMeshData;

        /** The input file stream. */
        std::ifstream instream;

        /** The tinyply file handle. */
        tinyply::PlyFile plf;

        /** Pointer to the opened file. */
        vislib::sys::File *file;

        /** The current data hash. */
        size_t data_hash;
    };

} /* end namespace io */
} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* MEGAMOL_DATATOOLS_IO_PLYDATASOURCE_H_INCLUDED */
