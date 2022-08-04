/*
 * ResourceWrapper.cpp
 *
 * Copyright (C) 2012 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#include "mmcore/utility/ResourceWrapper.h"
#include "vislib/sys/FastFile.h"
#include "vislib/sys/Path.h"

using namespace megamol::core;
using namespace megamol::core::utility;

/*
 * ResourceWrapper::getFileName
 */
vislib::StringW ResourceWrapper::getFileName(const Configuration& config, const vislib::StringA& name) {
    using megamol::core::utility::log::Log;

    vislib::StringW filename;
    const vislib::Array<vislib::StringW>& searchPaths = config.ResourceDirectories();
    for (SIZE_T i = 0; i < searchPaths.Count(); i++) {
        filename = vislib::sys::Path::Concatenate(searchPaths[i], vislib::StringW(name));
        if (vislib::sys::File::Exists(filename)) {
            Log::DefaultLog.WriteInfo("Loading %s ...\n", vislib::StringA(filename).PeekBuffer());
            break;
        } else {
            filename.Clear();
        }
    }
    return filename;
}


/*
 * ResourceWrapper::LoadResource
 */
SIZE_T ResourceWrapper::LoadResource(const Configuration& config, const vislib::StringA& name, void** outData) {
    using megamol::core::utility::log::Log;

    *outData = NULL;

    vislib::StringW filename = ResourceWrapper::getFileName(config, name);
    if (filename.IsEmpty()) {
        Log::DefaultLog.WriteError("Unable to load resource \"%s\": not found\n", name.PeekBuffer());
        return 0;
    }

    SIZE_T size = static_cast<SIZE_T>(vislib::sys::File::GetSize(filename));
    if (size < 1) {
        Log::DefaultLog.WriteError("Unable to load resource \"%s\": file is empty\n", name.PeekBuffer());
        return 0;
    }
    *outData = new BYTE[size];
    vislib::sys::FastFile f;
    if (!f.Open(filename, vislib::sys::File::READ_ONLY, vislib::sys::File::SHARE_READ, vislib::sys::File::OPEN_ONLY)) {
        Log::DefaultLog.WriteError("Unable to load resource \"%s\": cannot open file\n", name.PeekBuffer());
        ARY_SAFE_DELETE(*outData);
        return 0;
    }
    SIZE_T num = static_cast<SIZE_T>(f.Read(*outData, size));
    if (num != size) {
        Log::DefaultLog.WriteError("Unable to load resource \"%s\": cannot read whole file\n", name.PeekBuffer());
        ARY_SAFE_DELETE(*outData);
        return 0;
    }

    return num;
}


/*
 * ResourceWrapper::LoadTextResource
 */
SIZE_T ResourceWrapper::LoadTextResource(const Configuration& config, const vislib::StringA& name, char** outData) {
    using megamol::core::utility::log::Log;

    *outData = NULL;
    vislib::StringW filename = ResourceWrapper::getFileName(config, name);
    if (filename.IsEmpty()) {
        Log::DefaultLog.WriteError("Unable to load resource \"%s\": not found\n", name.PeekBuffer());
        return 0;
    }

    SIZE_T size = static_cast<SIZE_T>(vislib::sys::File::GetSize(filename));
    if (size < 1) {
        Log::DefaultLog.WriteError("Unable to load resource \"%s\": file is empty\n", name.PeekBuffer());
        return 0;
    }

    *outData = new char[size + 1];
    vislib::sys::FastFile f;
    if (!f.Open(filename, vislib::sys::File::READ_ONLY, vislib::sys::File::SHARE_READ, vislib::sys::File::OPEN_ONLY)) {
        Log::DefaultLog.WriteError("Unable to load resource \"%s\": cannot open file\n", name.PeekBuffer());
        ARY_SAFE_DELETE(*outData);
        return 0;
    }
    SIZE_T num = static_cast<SIZE_T>(f.Read(*outData, size));
    if (num != size) {
        Log::DefaultLog.WriteError("Unable to load resource \"%s\": cannot read whole file\n", name.PeekBuffer());
        ARY_SAFE_DELETE(*outData);
        return 0;
    }
    (*outData)[num] = 0;
    return num + 1;
}
