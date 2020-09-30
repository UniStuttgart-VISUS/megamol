/*
 * FileUtils.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "FileUtils.h"


using namespace megamol;
using namespace megamol::gui;


size_t megamol::gui::FileUtils::LoadRawFile(std::string name, void** outData) {

    *outData = nullptr;

    vislib::StringW filename = static_cast<vislib::StringW>(name.c_str());
    if (filename.IsEmpty()) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unable to load file: No file name given. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return 0;
    }

    if (!vislib::sys::File::Exists(filename)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unable to load file \"%s\": Not existing. [%s, %s, line %d]\n", name.c_str(), __FILE__, __FUNCTION__,
            __LINE__);
        return 0;
    }

    size_t size = static_cast<size_t>(vislib::sys::File::GetSize(filename));
    if (size < 1) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unable to load file \"%s\": File is empty. [%s, %s, line %d]\n", name.c_str(), __FILE__,
            __FUNCTION__, __LINE__);
        return 0;
    }

    vislib::sys::FastFile f;
    if (!f.Open(filename, vislib::sys::File::READ_ONLY, vislib::sys::File::SHARE_READ, vislib::sys::File::OPEN_ONLY)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unable to load file \"%s\": Unable to open file. [%s, %s, line %d]\n", name.c_str(), __FILE__,
            __FUNCTION__, __LINE__);
        return 0;
    }

    *outData = new BYTE[size];
    size_t num = static_cast<size_t>(f.Read(*outData, size));
    if (num != size) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unable to load file \"%s\": Unable to read whole file. [%s, %s, line %d]\n", name.c_str(), __FILE__,
            __FUNCTION__, __LINE__);
        ARY_SAFE_DELETE(*outData);
        return 0;
    }

    return num;
}


bool megamol::gui::FileUtils::WriteFile(const std::string& filename, const std::string& in_content) {
#ifdef GUI_USE_FILESYSTEM
    try {
        std::ofstream file;
        file.open(filename, std::ios_base::out);
        if (file.is_open() && file.good()) {
            file << in_content.c_str();
            file.close();
        } else {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Unable to create file. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            file.close();
            return false;
        }
    } catch (std::exception e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    return true;
#else
    return false;
#endif // GUI_USE_FILESYSTEM
}


bool megamol::gui::FileUtils::ReadFile(const std::string& filename, std::string& out_content) {
#ifdef GUI_USE_FILESYSTEM
    try {
        std::ifstream file;
        file.open(filename, std::ios_base::in);
        if (file.is_open() && file.good()) {
            out_content.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
            file.close();
        } else {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Unable to open file. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            file.close();
            return false;
        }
    } catch (std::exception e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    return true;
#else
    return false;
#endif // GUI_USE_FILESYSTEM
}
