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
#include "mmstd_datatools/GraphDataCall.h"
#include "vislib/sys/File.h"
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

        static const char *ClassName(void) { return "PLYDataSource"; }
        static const char *Description(void) { return "Data source module for .PLY files."; }
        static bool IsAvailable(void) { return true; }

        PLYDataSource(void);
        virtual ~PLYDataSource(void);


    protected:

        virtual bool create(void);
        virtual void release(void);

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

        bool filenameChanged(core::param::ParamSlot& slot);

        bool anyEnumChanged(core::param::ParamSlot& slot);

        bool getDataCallback(core::Call& caller);

        bool getExtentCallback(core::Call& caller);

        core::param::ParamSlot filename;

        core::param::ParamSlot vertElemSlot;
        core::param::ParamSlot faceElemSlot;

        core::param::ParamSlot xPropSlot;
        core::param::ParamSlot yPropSlot;
        core::param::ParamSlot zPropSlot;
        core::param::ParamSlot nxPropSlot;
        core::param::ParamSlot nyPropSlot;
        core::param::ParamSlot nzPropSlot;
        core::param::ParamSlot rPropSlot;
        core::param::ParamSlot gPropSlot;
        core::param::ParamSlot bPropSlot;
        core::param::ParamSlot iPropSlot;
        core::param::ParamSlot indexPropSlot;

        std::vector<std::string> guessedPos, guessedNormal, guessedColor;
        std::string guessedIndices;
        std::string guessedVertices, guessedFaces;

        /// this also guards data availability and consistency with element/property selection!
        std::shared_ptr<tinyply::PlyData> vertices;

        std::shared_ptr<tinyply::PlyData> normals, colors, faces;

        core::CalleeSlot getData;

        std::ifstream instream;
        tinyply::PlyFile plf;

        vislib::sys::File *file;
        size_t data_hash;

    };

} /* end namespace io */
} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* MEGAMOL_DATATOOLS_IO_PLYDATASOURCE_H_INCLUDED */
