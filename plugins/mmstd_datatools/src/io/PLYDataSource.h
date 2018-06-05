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
        bool getDataCallback(core::Call& caller);

        /**
         * Reimplementation of the getExtent callback.
         *
         * @param The triggering call.
         * @return True on success. False otherwise.
         */
        bool getExtentCallback(core::Call& caller);

        /** The slot for the filename */
        core::param::ParamSlot filename;
        
        /** The slot to put out the data */
        core::CalleeSlot getData;

        /** Pointer to the vislib file */
        vislib::sys::File *file;

        /** The current data hash */
        size_t data_hash;
    };

} /* end namespace io */
} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* MEGAMOL_DATATOOLS_IO_PLYDATASOURCE_H_INCLUDED */
