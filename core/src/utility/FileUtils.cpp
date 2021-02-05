/*
 * FileUtils.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/utility/FileUtils.h"


using namespace megamol::core::utility;


size_t megamol::core::utility::FileUtils::LoadRawFile(const std::wstring& filename, void** outData) {

    *outData = nullptr;

    auto file_name = static_cast<vislib::StringW>(filename.c_str());
    if (file_name.IsEmpty()) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unable to load file: No file name given. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return 0;
    }

    if (!vislib::sys::File::Exists(file_name)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unable to load file \"%s\": Not existing. [%s, %s, line %d]\n", filename.c_str(), __FILE__,
            __FUNCTION__,
            __LINE__);
        return 0;
    }

    size_t size = static_cast<size_t>(vislib::sys::File::GetSize(file_name));
    if (size < 1) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unable to load file \"%s\": File is empty. [%s, %s, line %d]\n", filename.c_str(), __FILE__,
            __FUNCTION__, __LINE__);
        return 0;
    }

    vislib::sys::FastFile f;
    if (!f.Open(file_name, vislib::sys::File::READ_ONLY, vislib::sys::File::SHARE_READ, vislib::sys::File::OPEN_ONLY)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unable to load file \"%s\": Unable to open file. [%s, %s, line %d]\n", filename.c_str(), __FILE__,
            __FUNCTION__, __LINE__);
        return 0;
    }

    *outData = new BYTE[size];
    size_t num = static_cast<size_t>(f.Read(*outData, size));
    if (num != size) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unable to load file \"%s\": Unable to read whole file. [%s, %s, line %d]\n", filename.c_str(),
            __FILE__,
            __FUNCTION__, __LINE__);
        ARY_SAFE_DELETE(*outData);
        return 0;
    }

    return num;
}


bool megamol::core::utility::FileUtils::WriteFile(const std::string& filename, const std::string& in_content, bool silent) {
    try {
        std::ofstream file;
        file.open(filename, std::ios_base::out);
        if (file.is_open() && file.good()) {
            file << in_content.c_str();
            file.close();
        } else {
            if (!silent)
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "[GUI] Unable to create file '%s'. [%s, %s, line %d]\n", filename.c_str(), __FILE__, __FUNCTION__,
                    __LINE__);
            file.close();

            return false;
        }
    } catch (std::exception& e) {
        if (!silent)
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (...) {
        if (!silent)
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    return true;
}


bool megamol::core::utility::FileUtils::ReadFile(const std::string& filename, std::string& out_content, bool silent) {
    try {
        std::ifstream file;
        file.open(filename, std::ios_base::in);
        if (file.is_open() && file.good()) {
            out_content.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
            file.close();
        } else {
            if (!silent)
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "[GUI] Unable to open file '%s'. [%s, %s, line %d]\n", filename.c_str(), __FILE__, __FUNCTION__,
                    __LINE__);
            file.close();
            return false;
        }
    } catch (std::exception& e) {
        if (!silent)
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (...) {
        if (!silent)
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    return true;
}
