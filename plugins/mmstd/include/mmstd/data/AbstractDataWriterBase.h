/*
 * AbstractDataWriterBase.h
 *
 * Copyright (C) 2010 by Universitaet Stuttgart (VISUS)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_ABSTRACTDATAWRITERBASE_H_INCLUDED
#define MEGAMOLCORE_ABSTRACTDATAWRITERBASE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/CallerSlot.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/utility/log/Log.h"
#include "mmstd/data/AbstractDataWriter.h"
#include "vislib/sys/FastFile.h"
#include "vislib/sys/Thread.h"
#include <climits>


namespace megamol {
namespace core {

/**
 * Abstract base class for data writer modules
 *
 * @param T The call to connect to the data source, must be derived from
 *          AbstractGetData3DCall(!) or must expose similar call functions
 * @param D The call description of T
 */
template<class T, class D = factories::CallAutoDescription<T>>
class AbstractDataWriterBase : public AbstractDataWriter {
public:
    /** Ctor. */
    AbstractDataWriterBase(void)
            : AbstractDataWriter()
            , inData("inData", "Get the data")
            , filenameSlot("filename", "The path to the MMPGD file to be written") {
        this->inData.template SetCompatibleCall<D>();
        this->MakeSlotAvailable(&this->inData);
        this->filenameSlot << new param::FilePathParam("");
        this->MakeSlotAvailable(&this->filenameSlot);
    }

    /** Dtor. */
    virtual ~AbstractDataWriterBase(void) {
        this->Release();
    }

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void) {
        // intentionally empty
        return true;
    }

    /**
     * Implementation of 'Release'.
     */
    virtual void release(void) {
        // intentionally empty
    }

    /**
     * The main function
     *
     * @return True on success
     */
    virtual bool run(void) {
        using megamol::core::utility::log::Log;
        vislib::TString filename(this->filenameSlot.template Param<param::FilePathParam>()->Value());
        if (filename.IsEmpty()) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "No file name specified. Abort.");
            return false;
        }

        T* d = this->inData.template CallAs<T>();
        if (d == NULL) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "No data source connected. Abort.");
            return false;
        }

        if (vislib::sys::File::Exists(filename)) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_WARN, "File %s already exists and will be overwritten.",
                vislib::StringA(filename).PeekBuffer());
        }

        // fetch extents
        if (!(*d)(1)) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Get extend failed.");
            return false;
        }

        vislib::sys::FastFile file;
        if (!file.Open(filename, vislib::sys::File::WRITE_ONLY, vislib::sys::File::SHARE_EXCLUSIVE,
                vislib::sys::File::CREATE_OVERWRITE)) {
            d->Unlock();
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to create output file \"%s\". Abort.",
                vislib::StringA(filename).PeekBuffer());
            return false;
        }

        this->writeFileStart(file, *d);

        unsigned int cnt = d->FrameCount();
        for (unsigned int idx = 0; idx < cnt; idx++) {
            unsigned int missCnt = 0;
            do {
                d->SetFrameID(idx, true);
                if (!(*d)(0)) {
                    Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Get data failed. Abort.");
                    missCnt = UINT_MAX;
                    break;
                }
                if (d->FrameID() != idx) {
                    d->Unlock();
                    missCnt++;
                    if (missCnt > 10) {
                        Log::DefaultLog.WriteMsg(
                            Log::LEVEL_ERROR + 50, "Unable to fetch requested frame %u. Abort.", idx);
                        break;
                    }
                    vislib::sys::Thread::Sleep(100);
                }
            } while (d->FrameID() != idx);
            if (missCnt > 10)
                break;

            this->writeFileFrameData(file, idx, *d);

            d->Unlock();
        }

        this->writeFileEnd(file);

        file.Close();

        return true;
    }

    /**
     * Function querying the writers capabilities
     *
     * @param call The call to receive the capabilities
     *
     * @return True on success
     */
    virtual bool getCapabilities(DataWriterCtrlCall& call) {
        call.SetAbortable(false); // for now
        return true;
    }

    /**
     * Starts writing the file
     *
     * @param file The file
     * @param data The data call after calling "GetExtent"
     */
    virtual void writeFileStart(vislib::sys::File& file, T& data) = 0;

    /**
     * Writes one data frame to the file. Frames are written from zero on
     * with continuously increasing frame numbers.
     *
     * @param file The file
     * @param idx The zero-based index of the current frame
     * @param data The data call after calling "GetData"
     */
    virtual void writeFileFrameData(vislib::sys::File& file, unsigned int idx, T& data) = 0;

    /**
     * Finishes writing the file
     *
     * @param file The file
     */
    virtual void writeFileEnd(vislib::sys::File& file) = 0;

private:
    /** The slot to get the data */
    CallerSlot inData;

    /** The file name of the file to be written */
    param::ParamSlot filenameSlot;
};

} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ABSTRACTDATAWRITER_H_INCLUDED */
