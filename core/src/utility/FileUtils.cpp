/*
 * FileUtils.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "mmcore/utility/FileUtils.h"


using namespace megamol::core::utility;


bool megamol::core::utility::FileUtils::WriteFile(
    const std::filesystem::path& filename, const std::string& in_content, bool silent) {

    try {
        std::ofstream file;
        file.open(filename);
        if (file.is_open() && file.good()) {
            file << in_content.c_str();
            file.close();
        } else {
            if (!silent)
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "Unable to create file '%s'. [%s, %s, line %d]\n", filename.c_str(), __FILE__, __FUNCTION__,
                    __LINE__);
            file.close();
            return false;
        }
    } catch (std::filesystem::filesystem_error& e) {
        if (!silent)
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Filesystem Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (std::exception& e) {
        if (!silent)
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (...) {
        if (!silent)
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    return true;
}


bool megamol::core::utility::FileUtils::ReadFile(
    const std::filesystem::path& filename, std::string& out_content, bool silent) {

    try {
        std::ifstream file;
        file.open(filename);
        if (file.is_open() && file.good()) {
            out_content.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
            file.close();
        } else {
            if (!silent)
                megamol::core::utility::log::Log::DefaultLog.WriteError("Unable to open file '%s'. [%s, %s, line %d]\n",
                    filename.c_str(), __FILE__, __FUNCTION__, __LINE__);
            file.close();
            return false;
        }
    } catch (std::filesystem::filesystem_error& e) {
        if (!silent)
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Filesystem Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (std::exception& e) {
        if (!silent)
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (...) {
        if (!silent)
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    return true;
}


bool megamol::core::utility::FileUtils::LoadRawFile(
    const std::filesystem::path& filename, std::vector<char>& out_data) {

    out_data.clear();
    try {
        if (filename.empty()) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Unable to load file: No file name given. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }

        if (!(std::filesystem::exists(filename) && std::filesystem::is_regular_file(filename))) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Unable to load file \"%s\": Not existing. [%s, %s, line %d]\n", filename.c_str(), __FILE__,
                __FUNCTION__, __LINE__);
            return false;
        }

        std::ifstream input_file(filename, std::ifstream::binary);
        if (!input_file.is_open() || !input_file.good()) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Unable to open file \"%s\": Bad file. [%s, %s, line %d]\n", filename.c_str(), __FILE__, __FUNCTION__,
                __LINE__);
            return false;
        }

        size_t size = 0;
        input_file.seekg(0, input_file.end);
        size = input_file.tellg();
        input_file.seekg(0, input_file.beg);
        if (size < 1) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Unable to load file \"%s\": File is empty. [%s, %s, line %d]\n", filename.c_str(), __FILE__,
                __FUNCTION__, __LINE__);
            return false;
        }

        out_data.resize(size);
        input_file.read(out_data.data(), size);
        if (!input_file.good()) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Unable to load file \"%s\": Unable to read whole file. [%s, %s, line %d]\n", filename.c_str(),
                __FILE__, __FUNCTION__, __LINE__);
            out_data.clear();
            return false;
        }
    } catch (std::filesystem::filesystem_error& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Filesystem Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (std::exception& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    return true;
}
