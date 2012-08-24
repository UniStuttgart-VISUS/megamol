/*
 * ResourceWrapper.cpp
 *
 * Copyright (C) 2012 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "ResourceWrapper.h"
#include "vislib/Path.h"
#include "vislib/BufferedFile.h"

using namespace megamol::core;
using namespace megamol::core::utility;

/*
 * ResourceWrapper::LoadResource
 */
SIZE_T ResourceWrapper::LoadResource(const Configuration& config, const vislib::StringA & name, void **outData) {
    using vislib::sys::Log;

    *outData = NULL;
    vislib::StringW filename;
    const vislib::Array<vislib::StringW>& searchPaths = config.ResourceDirectories();
    for (SIZE_T i = 0; i < searchPaths.Count(); i++) {
        filename = vislib::sys::Path::Concatenate(searchPaths[i], vislib::StringW(name));
        if (vislib::sys::File::Exists(filename)) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_INFO + 50,
                "Loading %s ...\n", vislib::StringA(filename).PeekBuffer());
            break;
        } else {
            filename.Clear();
        }
    }
    if (filename.IsEmpty()) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Unable to load resource \"%s\": not found\n",
            name.PeekBuffer());
        return 0;
    }

    SIZE_T size = vislib::sys::File::GetSize(filename);
    if (size < 1) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Unable to load resource \"%s\": file is empty\n",
            name.PeekBuffer());
        return 0;
    }
    *outData = new BYTE[size];
    vislib::sys::BufferedFile f;
    if (!f.Open(filename, vislib::sys::BufferedFile::READ_ONLY, vislib::sys::BufferedFile::SHARE_READ,
        vislib::sys::BufferedFile::OPEN_ONLY)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Unable to load resource \"%s\": cannot open file\n",
            name.PeekBuffer());
        ARY_SAFE_DELETE(*outData);
        return 0;
    }
    SIZE_T num = f.Read(*outData, size);
    if (num != size) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Unable to load resource \"%s\": cannot read whole file\n",
            name.PeekBuffer());
        ARY_SAFE_DELETE(*outData);
        return 0;
    }
    
    return num;
}
