#pragma once

#include <chrono>
#include <iomanip>
#include <sstream>
#include <string>

#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/ParamSlot.h"

#include "vislib/sys/DirectoryIterator.h"

namespace megamol::core::utility {
enum class increment_type : uint8_t { NONE, INCREMENT_SAFE, INCREMENT_OVERWRITE, TIMESTAMP, TIMESTEP };

inline std::string get_extended_filename(
    param::ParamSlot& filename_slot, std::size_t const frame_id, std::size_t& increment, increment_type inc_type) {
    // Get filename
    const auto& vislib_filename = filename_slot.template Param<core::param::FilePathParam>()->Value();
    std::string filename(vislib_filename.PeekBuffer());

    // Modify file name suffix according to user's selection
    const std::string leading_zeroes("0000000000");
    std::string suffix("_");

    if (inc_type == increment_type::INCREMENT_SAFE) {
        // Increment safely
        if (filename_slot.IsDirty()) {
            filename_slot.ResetDirty();

            // Find file with largest suffix number
            bool found = false;
            increment = 0;

            const std::string directory_path = filename.substr(0, filename.find_last_of("/\\"));
            vislib::sys::DirectoryIteratorA directory(directory_path.c_str(), false, false);

            while (directory.HasNext()) {
                const auto entry = directory.Next();
                const std::string path(entry.Path.PeekBuffer(), entry.Path.Length());

                std::string extracted_suffix = path.substr(0, path.find_last_of('.'));
                extracted_suffix = extracted_suffix.substr(extracted_suffix.find_last_of('_') + 1);

                if (extracted_suffix.length() >= leading_zeroes.length() &&
                    extracted_suffix.find_first_not_of("0123456789") == std::string::npos) {
                    increment = std::max(increment, static_cast<std::size_t>(std::stoull(extracted_suffix)));
                    found = true;
                }
            }

            // If a file exists, take the next available number
            if (found) {
                ++increment;
            }
        }

        const std::string number = std::to_string(increment++);
        const std::string number_with_lead = leading_zeroes + number;
        suffix += number_with_lead.substr(std::min(number.length(), leading_zeroes.length()));
    } else if (inc_type == increment_type::INCREMENT_OVERWRITE) {
        // Increment beginning at 0
        if (filename_slot.IsDirty()) {
            filename_slot.ResetDirty();
            increment = 0;
        }

        const std::string number = std::to_string(increment++);
        const std::string number_with_lead = leading_zeroes + number;
        suffix += number_with_lead.substr(std::min(number.length(), leading_zeroes.length()));
    } else if (inc_type == increment_type::TIMESTAMP) {
        // Time stamp
        std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
        std::time_t now_c = std::chrono::system_clock::to_time_t(now);

        std::stringstream ss;
        ss << std::put_time(std::localtime(&now_c), "%Y-%m-%d_%H-%M-%S") << "_";
        ss << std::chrono::duration_cast<std::chrono::milliseconds>(now - std::chrono::system_clock::from_time_t(now_c))
                  .count();

        suffix += ss.str();
    } else if (inc_type == increment_type::TIMESTEP) {
        // Current time step
        const std::string number = std::to_string(frame_id);
        const std::string number_with_lead = leading_zeroes + number;
        suffix += number_with_lead.substr(std::min(number.length(), leading_zeroes.length()));
    }

    if (suffix.length() > 1) {
        const auto dot_pos = filename.find_last_of('.');
        filename = filename.substr(0, dot_pos) + suffix + filename.substr(dot_pos);
    }

    return filename;
}
} // namespace megamol::core::utility
