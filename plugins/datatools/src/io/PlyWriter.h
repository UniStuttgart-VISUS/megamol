/*
 * PlyWriter.h
 *
 * Copyright (C) 2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#ifndef MEGAMOL_DATATOOLS_IO_PLYWRITER_H_INCLUDED
#define MEGAMOL_DATATOOLS_IO_PLYWRITER_H_INCLUDED
#pragma once

#include "mmcore/AbstractDataWriter.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "geometry_calls/CallTriMeshData.h"
#include "vislib/sys/FastFile.h"

namespace megamol {
namespace stdplugin {
namespace datatools {
namespace io {

    class PlyWriter : public core::AbstractDataWriter {
    public:
    
        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "PlyWriter";
        }
    
        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Polygon file format (.ply) file writer";
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
         * Disallow usage in quickstarts
         *
         * @return false
         */
        static bool SupportQuickstart(void) {
            return false;
        }
    
        /** Ctor. */
        PlyWriter(void);
    
        /** Dtor. */
        virtual ~PlyWriter(void);
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
         * The main function
         *
         * @return True on success
         */
        virtual bool run(void);
    
        /**
         * Function querying the writers capabilities
         *
         * @param call The call to receive the capabilities
         *
         * @return True on success
         */
        virtual bool getCapabilities(core::DataWriterCtrlCall& call);
    
    private:
    
        /** The file name of the file to be written */
        core::param::ParamSlot filenameSlot;
    
        /** The frame ID of the frame to be written */
        core::param::ParamSlot frameIDSlot;
    
        /** The slot asking for data. */
        core::CallerSlot meshDataSlot;
    };

} /* end namespace io */
} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif // !MEGAMOL_DATATOOLS_IO_PLYWRITER_H_INCLUDED