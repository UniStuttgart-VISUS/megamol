/*
 * AbstractSTLWriter.h
 *
 * Copyright (C) 2018 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#ifndef MEGAMOL_DATATOOLS_IO_ABSTRACTSTLWRITER_H_INCLUDED
#define MEGAMOL_DATATOOLS_IO_ABSTRACTSTLWRITER_H_INCLUDED
#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmstd/data/AbstractGetData3DCall.h"

#include "mmcore/factories/CallDescription.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/ParamSlot.h"

#include "mmcore/utility/log/Log.h"
#include "vislib/sys/DirectoryEntry.h"
#include "vislib/sys/DirectoryIterator.h"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <ios>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "AbstractSTLWriter.aux"

namespace megamol::datatools::io {
/// <summary>
/// Abstract writer for STL files.
///
/// This class provides a writer for STL files that is connected in between a data source and a data sink with the same call type.
/// Information from the sink is forwarded to the source, as well as the other way round. Additionally, the data is copied
/// from the source to the sink. This way, this module can automatically write data to file whenever the input changes or a new
/// frame is requested by the sink.
///
/// All necessary input and output slots are already provided. For them to work, the class accepts a template parameter
/// InputCallDescription that provides the information about the compatible call. Additionally, the derived class has to call
/// the constructor with the name of the call, as needed for setting the callback functions accordingly.
///
/// Slots for parameters are also provided, which control the write operation:
/// - File name            - Path and file name of the file to be written (default: empty)
/// - ASCII/Binary option  - Option to choose between ASCII and binary output (default: binary)
/// - Automatic writing    - Checkbox to allow automatic writing when new data is provided by the source (default: on)
/// - File name increment  - Option to choose how to behave in naming and when the output file already exists (default: increment)
///                         None:                          Overwrite file if it exists
///                          Increment number (safe):       Append a number to prevent overwriting
///                          Increment number (overwrite):  Start with 0 and keep incrementing to prevent overwriting previously created files
///                          Time stamp:                    Append human-readable, sortable time stamp
///                          Current time step:             Append number representing the current time step, as set in the call
/// - Write now button     - Button to execute the write command immediately on the data of the last call, if possible
///
/// On extent callback:
/// - Information from the AbstractGetData3DCall is copied from the sink to the source
/// - The user is asked to copy further information, that is not part of the AbstractGetData3DCall, to the source
/// - The outgoing call to the source is made to provide the extent
/// - Information from the AbstractGetData3DCall is copied from the source to the sink
/// - The user is asked to copy further information, that is not part of the AbstractGetData3DCall, to the sink
///
/// On data callback:
/// - The outgoing call to the source is made to provide the data
/// - The hash is compared and if it differs from the one in the sink:
///   - The user is asked to copy the new data from the source to the sink
///   - The user is asked to prepare the data and call the write function of this class
///
/// The following functions have to be overridden, additionally to create and release inherited from Module:
/// - copy_info_upstream(...)    - Copy information from sink to source
/// - copy_info_downstream(...)  - Copy information from source to sink
/// - copy_data(...)             - Copy data from source to sink
/// - write_data(...)            - Prepare data and call the write function
///
/// For an example implementation, see TriMeshSTLWriter.
/// </summary>
/// <typeparam name="InputCallDescription">Description of the incoming and outgoing call used to set as compatible call</typeparam>
template<typename InputCallDescription>
class AbstractSTLWriter : public core::Module {
public:
    /// <summary>
    /// Constructor
    /// </summary>
    /// <param name="call_name">Name of the connected call</param>
    AbstractSTLWriter(const char* call_name)
            : core::Module()
            , last_callee(nullptr)
            , input_slot("in_mesh_data", "Input triangle mesh data")
            , output_slot("out_mesh_data", "Output triangle mesh data")
            , filename_slot("STL file", "The name of to the STL file to write")
            , ascii_binary_slot("Output type", "Write an ASCII or binary file?")
            , write_automatically("Write automatically", "Write automatically when underlying data has changed")
            , filename_increment("File name suffix", "Increment the file name suffix to prevent overwriting files")
            , write_now("Write now", "Write an STL file immediately") {
        // Create input slot for triangle mesh data
        this->input_slot.template SetCompatibleCall<InputCallDescription>();
        Module::MakeSlotAvailable(&this->input_slot);

        // Create output information
        this->output_slot.SetCallback(
            call_name, "GetExtent", &AbstractSTLWriter<InputCallDescription>::get_extent_callback);
        this->output_slot.SetCallback(
            call_name, "GetData", &AbstractSTLWriter<InputCallDescription>::get_data_callback);

        Module::MakeSlotAvailable(&this->output_slot);

        // Create file name textbox
        this->filename_slot << new core::param::FilePathParam("");
        Module::MakeSlotAvailable(&this->filename_slot);

        // Create enum for ASCII/binary option
        this->ascii_binary_slot << new core::param::EnumParam(0);
        this->ascii_binary_slot.template Param<core::param::EnumParam>()->SetTypePair(0, "Binary");
        this->ascii_binary_slot.template Param<core::param::EnumParam>()->SetTypePair(1, "ASCII");
        Module::MakeSlotAvailable(&this->ascii_binary_slot);

        // Write automatically?
        this->write_automatically << new core::param::BoolParam(true);
        Module::MakeSlotAvailable(&this->write_automatically);

        // Use file name suffix increment?
        this->filename_increment << new core::param::EnumParam(1);
        this->filename_increment.template Param<core::param::EnumParam>()->SetTypePair(0, "None (overwrite)");
        this->filename_increment.template Param<core::param::EnumParam>()->SetTypePair(1, "Increment number (safe)");
        this->filename_increment.template Param<core::param::EnumParam>()->SetTypePair(
            2, "Increment number (overwrite)");
        this->filename_increment.template Param<core::param::EnumParam>()->SetTypePair(3, "Time stamp");
        this->filename_increment.template Param<core::param::EnumParam>()->SetTypePair(4, "Current time step");
        Module::MakeSlotAvailable(&this->filename_increment);

        // Button to write file manually
        this->write_now << new core::param::ButtonParam();
        this->write_now.SetUpdateCallback(&AbstractSTLWriter::write_manually_callback);
        Module::MakeSlotAvailable(&this->write_now);
    }

    /// <summary>
    /// Destructor
    /// </summary>
    ~AbstractSTLWriter() override {}

protected:
    /// <summary>
    /// Create the module
    /// </summary>
    /// <returns>True on success; false otherwise</returns>
    bool create() override = 0;

    /// <summary>
    /// Copy information from the incoming to the outgoing call
    /// </summary>
    /// <param name="caller">Incoming call</param>
    /// <param name="callee">Outgoing call</param>
    /// <returns>True on success; false otherwise</returns>
    virtual bool copy_info_upstream(core::AbstractGetData3DCall& caller, core::AbstractGetData3DCall& callee) = 0;

    /// <summary>
    /// Copy information from the outgoing to the incoming call
    /// </summary>
    /// <param name="caller">Incoming call</param>
    /// <param name="callee">Outgoing call</param>
    /// <returns>True on success; false otherwise</returns>
    virtual bool copy_info_downstream(core::AbstractGetData3DCall& caller, core::AbstractGetData3DCall& callee) = 0;

    /// <summary>
    /// Copy data to incoming call
    /// </summary>
    /// <param name="caller">Incoming call</param>
    /// <param name="callee">Outgoing call</param>
    /// <returns>True on success; false otherwise</returns>
    virtual bool copy_data(core::AbstractGetData3DCall& caller, core::AbstractGetData3DCall& callee) = 0;

    /// <summary>
    /// Write data from outgoing call to file
    /// </summary>
    /// <param name="callee">Outgoing call</param>
    /// <returns>True on success; false otherwise</returns>
    virtual bool write_data(core::AbstractGetData3DCall& callee) = 0;

    /// <summary>
    /// Release the module
    /// </summary>
    void release() override = 0;

    /// <summary>
    /// Callback function for requesting information
    /// </summary>
    /// <param name="caller">Call for this request</param>
    /// <returns>True on success; false otherwise</returns>
    bool get_extent_callback(core::Call& caller) {
        // Create outgoing call
        auto& incoming_call = dynamic_cast<core::AbstractGetData3DCall&>(caller);
        auto& outgoing_call = *this->input_slot.template CallAs<core::AbstractGetData3DCall>();

        // Copy incoming call information to outgoing
        outgoing_call.AccessBoundingBoxes() = incoming_call.GetBoundingBoxes();
        outgoing_call.SetFrameCount(incoming_call.FrameCount());
        outgoing_call.SetFrameID(incoming_call.FrameID(), incoming_call.IsFrameForced());

        if (!copy_info_upstream(incoming_call, outgoing_call)) {
            return false;
        }

        // Call for get extents
        if (!outgoing_call(1)) {
            return false;
        }

        // Copy outgoing call information to incoming
        incoming_call.AccessBoundingBoxes() = outgoing_call.GetBoundingBoxes();
        incoming_call.SetFrameCount(outgoing_call.FrameCount());
        incoming_call.SetFrameID(outgoing_call.FrameID(), incoming_call.IsFrameForced());

        if (!copy_info_downstream(incoming_call, outgoing_call)) {
            return false;
        }

        return true;
    }

    /// <summary>
    /// Callback function for requesting data
    /// </summary>
    /// <param name="caller">Call for this request</param>
    /// <returns>True on success; false otherwise</returns>
    bool get_data_callback(core::Call& caller) {
        // Create outgoing call
        auto& incoming_call = dynamic_cast<core::AbstractGetData3DCall&>(caller);
        auto& outgoing_call = *this->input_slot.template CallAs<core::AbstractGetData3DCall>();

        // Call for get data
        if (!outgoing_call(0)) {
            return false;
        }

        if (outgoing_call.DataHash() != incoming_call.DataHash()) {
            incoming_call.SetDataHash(outgoing_call.DataHash());

            // Copy data from outgoing to incoming call
            if (!copy_data(incoming_call, outgoing_call)) {
                return false;
            }

            // Write data from outgoing call to file
            if (this->write_automatically.template Param<core::param::BoolParam>()->Value()) {
                if (!write_data(outgoing_call)) {
                    return false;
                }
            }

            this->last_callee = &outgoing_call;
        }

        return true;
    }

    /// <summary>
    /// Callback function for writing file immediately
    /// </summary>
    /// <param name="slot">Parameter slot that triggered the function</param>
    /// <returns>True on success; false otherwise</returns>
    bool write_manually_callback(core::param::ParamSlot&) {
        // Write data to file if possible
        if (this->last_callee != nullptr) {
            if (!write_data(*this->last_callee)) {
                return false;
            }
        }

        return true;
    }

    /// <summary>
    /// Write an STL file
    /// </summary>
    /// <param name="num_triangles">Number of triangles</param>
    /// <param name="frame_id">Current frame ID</param>
    /// <param name="vertices">Vertex pointer</param>
    /// <param name="normals">Normal pointer</param>
    /// <param name="indices">Index pointer</param>
    /// <typeparam name="VFT">Floating point type of the vertices</typeparam>
    /// <typeparam name="NFT">Floating point type of the normals</typeparam>
    /// <typeparam name="IT">Integer type of the indices</typeparam>
    template<typename VFT, typename NFT, typename IT = std::nullptr_t>
    void write(const std::size_t num_triangles, const std::size_t frame_id, const VFT* vertices, const NFT* normals,
        const IT* indices = nullptr) {
        static_assert(
            std::is_integral<IT>::value || std::is_null_pointer<IT>::value, "Indices must be of integral type");

        // Get filename
        const auto& vislib_filename = this->filename_slot.template Param<core::param::FilePathParam>()->Value();
        std::string filename(vislib_filename.string());

        // Modify file name suffix according to user's selection
        const std::string leading_zeroes("0000000000");
        std::string suffix("_");

        if (this->filename_increment.template Param<core::param::EnumParam>()->Value() == 1) {
            // Increment safely
            if (this->filename_slot.IsDirty()) {
                this->filename_slot.ResetDirty();

                // Find file with largest suffix number
                bool found = false;
                this->increment = 0;

                const std::string directory_path = filename.substr(0, filename.find_last_of("/\\"));
                vislib::sys::DirectoryIteratorA directory(directory_path.c_str(), false, false);

                while (directory.HasNext()) {
                    const auto entry = directory.Next();
                    const std::string path(entry.Path.PeekBuffer(), entry.Path.Length());

                    std::string extracted_suffix = path.substr(0, path.find_last_of('.'));
                    extracted_suffix = extracted_suffix.substr(extracted_suffix.find_last_of('_') + 1);

                    if (extracted_suffix.length() >= leading_zeroes.length() &&
                        extracted_suffix.find_first_not_of("0123456789") == std::string::npos) {
                        this->increment =
                            std::max(this->increment, static_cast<std::size_t>(std::stoull(extracted_suffix)));
                        found = true;
                    }
                }

                // If a file exists, take the next available number
                if (found) {
                    ++this->increment;
                }
            }

            const std::string number = std::to_string(this->increment++);
            const std::string number_with_lead = leading_zeroes + number;
            suffix += number_with_lead.substr(std::min(number.length(), leading_zeroes.length()));
        } else if (this->filename_increment.template Param<core::param::EnumParam>()->Value() == 2) {
            // Increment beginning at 0
            if (this->filename_slot.IsDirty()) {
                this->filename_slot.ResetDirty();
                this->increment = 0;
            }

            const std::string number = std::to_string(this->increment++);
            const std::string number_with_lead = leading_zeroes + number;
            suffix += number_with_lead.substr(std::min(number.length(), leading_zeroes.length()));
        } else if (this->filename_increment.template Param<core::param::EnumParam>()->Value() == 3) {
            // Time stamp
            std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
            std::time_t now_c = std::chrono::system_clock::to_time_t(now);

            std::stringstream ss;
            ss << std::put_time(std::localtime(&now_c), "%Y-%m-%d_%H-%M-%S") << "_";
            ss << std::chrono::duration_cast<std::chrono::milliseconds>(
                now - std::chrono::system_clock::from_time_t(now_c))
                      .count();

            suffix += ss.str();
        } else if (this->filename_increment.template Param<core::param::EnumParam>()->Value() == 4) {
            // Current time step
            const std::string number = std::to_string(frame_id);
            const std::string number_with_lead = leading_zeroes + number;
            suffix += number_with_lead.substr(std::min(number.length(), leading_zeroes.length()));
        }

        if (suffix.length() > 1) {
            const auto dot_pos = filename.find_last_of('.');
            filename = filename.substr(0, dot_pos) + suffix + filename.substr(dot_pos);
        }

        // Sanity checks
        if (num_triangles == 0) {
            megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                "Cannot write STL file. Number of triangles is zero");
            return;
        }

        static_assert(std::is_floating_point<VFT>::value, "Vertices must be of floating point type");
        static_assert(std::is_floating_point<NFT>::value, "Normals must be of floating point type");

        if (vertices == nullptr || normals == nullptr) {
            megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                "Cannot write STL file. No vertices or normals given");
            return;
        }

        // Decide file type
        if (this->ascii_binary_slot.template Param<core::param::EnumParam>()->Value() == 0) {
            megamol::core::utility::log::Log::DefaultLog.WriteInfo("Writing binary STL file '%s'", filename.c_str());
            write_binary(
                filename, static_cast<uint32_t>(num_triangles), vertices, normals, pointer_or_identity<IT>(indices));
        } else {
            megamol::core::utility::log::Log::DefaultLog.WriteInfo("Writing ASCII STL file '%s'", filename.c_str());
            write_ascii(filename, num_triangles, vertices, normals, pointer_or_identity<IT>(indices));
        }
    }

private:
    /// <summary>
    /// Write a binary file
    /// </summary>
    /// <param name="filename">File name of the STL file</param>
    /// <param name="num_triangles">Number of triangles</param>
    /// <param name="vertices_ptr">Vertex pointer</param>
    /// <param name="normals_ptr">Normal pointer</param>
    /// <param name="indices">Index pointer</param>
    /// <typeparam name="VFT">Floating point type of the vertices</typeparam>
    /// <typeparam name="NFT">Floating point type of the normals</typeparam>
    /// <typeparam name="IT">Integer type of the indices</typeparam>
    template<typename VFT, typename NFT, typename IT>
    void write_binary(const std::string& filename, const uint32_t num_triangles, const VFT* vertices_ptr,
        const NFT* normals_ptr, const IT& indices) const {
        // Convert vertices and normals to float if it is not
        auto vertices_wrapper = convert_if_necessary<float>(vertices_ptr, num_triangles * 9);
        auto normals_wrapper = convert_if_necessary<float>(normals_ptr, num_triangles * 9);

        const float* vertices = vertices_wrapper.get();
        const float* normals = normals_wrapper.get();

        // Open or create file
        std::ofstream ofs(filename, std::ios_base::out | std::ios_base::binary);

        if (ofs.good()) {
            // Write header
            std::string header_message("MegaMol by University of Stuttgart, Germany");

            std::vector<char> header_buffer(header_message.begin(), header_message.end());
            header_buffer.resize(80);

            ofs.write(header_buffer.data(), header_buffer.size());

            // Write number of triangles
            ofs.write(reinterpret_cast<const char*>(&num_triangles), sizeof(uint32_t));

            // Write vertices and normals
            const uint16_t additional_attribute = 21313;

            for (uint32_t triangle_index = 0; triangle_index < num_triangles; ++triangle_index) {
                ofs.write(reinterpret_cast<const char*>(&normals[indices[triangle_index * 3] * 3]), 3 * sizeof(float));
                ofs.write(
                    reinterpret_cast<const char*>(&vertices[indices[triangle_index * 3 + 0] * 3]), 3 * sizeof(float));
                ofs.write(
                    reinterpret_cast<const char*>(&vertices[indices[triangle_index * 3 + 1] * 3]), 3 * sizeof(float));
                ofs.write(
                    reinterpret_cast<const char*>(&vertices[indices[triangle_index * 3 + 2] * 3]), 3 * sizeof(float));
                ofs.write(reinterpret_cast<const char*>(&additional_attribute), sizeof(uint16_t));
            }

            ofs.close();
        } else {
            std::stringstream ss;
            ss << "Binary STL file '" << filename << "' could not be written";

            throw std::runtime_error(ss.str());
        }
    }

    /// <summary>
    /// Write an ASCII file
    /// </summary>
    /// <param name="filename">File name of the STL file</param>
    /// <param name="num_triangles">Number of triangles</param>
    /// <param name="vertices">Vertex pointer</param>
    /// <param name="normals">Normal pointer</param>
    /// <param name="indices">Index pointer</param>
    /// <typeparam name="VFT">Floating point type of the vertices</typeparam>
    /// <typeparam name="NFT">Floating point type of the normals</typeparam>
    /// <typeparam name="IT">Integer type of the indices</typeparam>
    template<typename VFT, typename NFT, typename IT>
    void write_ascii(const std::string& filename, const std::size_t num_triangles, const VFT* vertices,
        const NFT* normals, const IT& indices) const {
        // Open or create file
        std::ofstream ofs(filename, std::ios_base::out);

        if (ofs.good()) {
            // Write mesh
            ofs << "solid megamol_mesh\n";
            ofs << std::scientific;

            for (std::size_t triangle_index = 0; triangle_index < num_triangles; ++triangle_index) {
                ofs << "\tfacet\n";

                ofs << "\t\tnormal " << normals[indices[triangle_index * 3] * 3 + 0] << " "
                    << normals[indices[triangle_index * 3] * 3 + 1] << " "
                    << normals[indices[triangle_index * 3] * 3 + 2] << "\n";

                ofs << "\t\touter loop\n";

                ofs << "\t\t\tvertex " << vertices[indices[triangle_index * 3 + 0] * 3 + 0] << " "
                    << vertices[indices[triangle_index * 3 + 0] * 3 + 1] << " "
                    << vertices[indices[triangle_index * 3 + 0] * 3 + 2] << "\n";
                ofs << "\t\t\tvertex " << vertices[indices[triangle_index * 3 + 1] * 3 + 0] << " "
                    << vertices[indices[triangle_index * 3 + 1] * 3 + 1] << " "
                    << vertices[indices[triangle_index * 3 + 1] * 3 + 2] << "\n";
                ofs << "\t\t\tvertex " << vertices[indices[triangle_index * 3 + 2] * 3 + 0] << " "
                    << vertices[indices[triangle_index * 3 + 2] * 3 + 1] << " "
                    << vertices[indices[triangle_index * 3 + 2] * 3 + 2] << "\n";

                ofs << "\t\tendloop\n";

                ofs << "\tendfacet" << std::endl;
            }

            ofs << "endsolid megamol_mesh" << std::flush;

            ofs.close();
        } else {
            std::stringstream ss;
            ss << "ASCII STL file '" << filename << "' could not be written";

            throw std::runtime_error(ss.str());
        }
    }

    /// Input
    core::CallerSlot input_slot;

    /// Output
    core::CalleeSlot output_slot;

    /// Last outgoing call
    core::AbstractGetData3DCall* last_callee;

    /// File name
    core::param::ParamSlot filename_slot;

    /// Option for ASCII/binary
    core::param::ParamSlot ascii_binary_slot;

    /// Write automatically?
    core::param::ParamSlot write_automatically;

    /// Use incrementing file name suffix when file already exists?
    core::param::ParamSlot filename_increment;

    /// Button for executing writing immediately
    core::param::ParamSlot write_now;

    /// Increment for file name suffix
    std::size_t increment;
};
} // namespace megamol::datatools::io

#endif // !MEGAMOL_DATATOOLS_IO_ABSTRACTSTLWRITER_H_INCLUDED
