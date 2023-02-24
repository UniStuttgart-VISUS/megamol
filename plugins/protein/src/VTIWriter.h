//
// VTIWriter.h
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: Apr 19, 2013
//     Author: scharnkn
//

#pragma once

#include "HostArr.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/job/AbstractJob.h"
#include "mmcore/param/ParamSlot.h"
#include "protein_calls/VTIDataCall.h"

namespace megamol::protein {

class VTIWriter : public core::job::AbstractJob, public core::Module {

public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "VTIWriter";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Writer module for *.vti file format used by the Visualization \
                Toolkit.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    /**
     * Ctor
     */
    VTIWriter();

    /**
     * Dtor
     */
    ~VTIWriter() override;

    /**
     * Answers whether or not this job is still running.
     *
     * @return 'true' if this job is still running, 'false' if it has
     *         finished.
     */
    bool IsRunning() const override;

    /**
     * Starts the job thread.
     *
     * @return true if the job has been successfully started.
     */
    bool Start() override;

    /**
     * Terminates the job thread.
     *
     * @return true to acknowledge that the job will finish as soon
     *         as possible, false if termination is not possible.
     */
    bool Terminate() override;

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

    /**
     * Implementation of 'Release'.
     */
    void release() override;

private:
    /**
     * Creates the folders in the output path that do not yet exist.
     *
     * @return 'True' on success, 'false' otherwise
     */
    bool createDirectories(vislib::StringA folder);

    /**
     * TODO
     */
    vislib::TString getFormatStr(protein_calls::VTKImageData::DataFormat f);

    /**
     * TODO
     */
    vislib::TString getTypeStr(protein_calls::VTKImageData::DataArray::DataType t);

    /**
     * TODO
     */
    bool writeDataAscii(
        const void* data, size_t size, std::ofstream& outfile, protein_calls::VTKImageData::DataArray::DataType t);

    /**
     * TODO
     */
    bool writeDataBinary(
        const void* data, size_t size, std::ofstream& outfile, protein_calls::VTKImageData::DataArray::DataType t);

    /**
     * TODO
     */
    bool writeDataAsciiFloat(const float* data, size_t size, std::ofstream& outfile);

    /**
     * TODO
     */
    bool writeDataBinaryFloat(const float* data, size_t size, std::ofstream& outfile);

    /**
     * TODO
     */
    bool writeDataArray(const protein_calls::VTIDataCall* dc, bool isPointData, unsigned int dataIdx,
        unsigned int pieceIdx, std::ofstream& outfile);

    /**
     * TODO
     */
    bool writeFile(protein_calls::VTIDataCall* dc);

    /**
     * TODO
     */
    bool writePiece(const protein_calls::VTIDataCall* dc, uint idx, std::ofstream& outfile);

    /// Data caller slot
    core::CallerSlot dataCallerSlot;

    /// Parameter to determine the first frame to be written
    core::param::ParamSlot minFrameSlot;

    /// Parameter to determine the number of frames to be written
    core::param::ParamSlot nFramesSlot;

    /// Parameter to determine the stride used when writing frames
    core::param::ParamSlot strideSlot;

    /// Parameter for the filename prefix
    core::param::ParamSlot filenamePrefixSlot;

    /// Parameter for the output folder
    core::param::ParamSlot outDirSlot;

    /// Parameter for the output format of the data
    core::param::ParamSlot dataFormatSlot;

    bool jobDone;       ///> Flag whether the job is done
    int filenameDigits; ///> Number of digits used in generated filenames

    HostArr<char> buffEn;  ///> Temp buffer used to encode data
    HostArr<char> buffDec; ///> Temp buffer used to hold decoded data
};

} // namespace megamol::protein
