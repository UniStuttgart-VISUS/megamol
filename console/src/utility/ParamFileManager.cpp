/*
 * ParamFileManager.cpp
 *
 * Copyright (C) 2016 MegaMol Team
 * Alle Rechte vorbehalten. All rights reserved.
 */
#include "stdafx.h"
#include "ParamFileManager.h"
#include "mmcore/api/MegaMolCore.h"
#include "CoreHandle.h"
#include "vislib/sys/Path.h"
#include "vislib/Array.h"
#include "vislib/StringTokeniser.h"
#include "vislib/sys/BufferedFile.h"
#include "vislib/sys/Log.h"
#include "vislib/sys/sysfunctions.h"
#include "vislib/sys/SystemException.h"
#include <string>
#include <utility>
#include <vector>

using namespace megamol;
using namespace megamol::console;

utility::ParamFileManager& utility::ParamFileManager::Instance() {
    static ParamFileManager inst;
    return inst;
}

namespace {

    struct ctxt {
        vislib::TString filename;
        bool useRelativeFileNames;
        void *hCore;
        std::vector<std::pair<std::string, std::string> > pf;
        std::vector<std::string> comments;
    };

    template<class T>
    static bool stringFSEqual(const T& lhs, const T& rhs) {
    #ifdef _WIN32
        return lhs.Equals(rhs, false);
    #else /* _WIN32 */
        return lhs.Equals(rhs, true);
    #endif /* _WIN32 */
    }

    static vislib::TString relativePathTo(const vislib::TString& path,
        const vislib::TString& base) {
        TCHAR pathSep = vislib::TString(vislib::sys::Path::SEPARATOR_A, 1)[0];
        vislib::Array<vislib::TString> pa = vislib::TStringTokeniser::Split(path, pathSep, false);
        vislib::Array<vislib::TString> ba = vislib::TStringTokeniser::Split(base, pathSep, false);

        if ((pa.Count() <= 0) || (ba.Count() <= 0)) return path;
        if (!stringFSEqual(pa[0], ba[0])) return path;

        SIZE_T i = 1;
        while ((i < pa.Count()) && (i < ba.Count()) && stringFSEqual(pa[i], ba[i])) ++i; // number of similar levels.

        vislib::TString result;
        for (SIZE_T j = 0; j < ba.Count() - i; ++j) {
            if (j > 0) result += pathSep;
            result += _T("..");
        }
        for (; i < pa.Count(); ++i) {
            result += pathSep;
            result += pa[i];
        }

        return result;
    }

    void enumParameter(const char *str, struct ctxt *c) {
        CoreHandle hParam;

        if (!::mmcGetParameterA(c->hCore, str, hParam)) {
            std::string s = std::string("Failed to get handle for parameter ") + std::string(str);
            c->comments.push_back(s);
            return;
        }

        std::string name(str);

        unsigned int len = 0;
        unsigned char *buf = NULL;
        ::mmcGetParameterTypeDescription(hParam, NULL, &len);
        buf = new unsigned char[len];
        ::mmcGetParameterTypeDescription(hParam, buf, &len);
        if ((len >= 6) && (::memcmp(buf, "MMBUTN", 6) == 0)) {
            name = std::string("# ") + name;
        }

        std::string value(::mmcGetParameterValueA(hParam));

        if (c->useRelativeFileNames && (len >= 6) && (
                (::memcmp(buf, "MMSTRW", 6) == 0) || (::memcmp(buf, "MMSTRA", 6) == 0)
                || (::memcmp(buf, "MMFILW", 6) == 0) || (::memcmp(buf, "MMFILA", 6) == 0))) {

            // fetch value
            vislib::TString v = ::mmcGetParameterValue(hParam);

            if (vislib::sys::Path::IsAbsolute(v)) {
                // value seems like a legal path
                vislib::TString paramFileDir = vislib::sys::Path::GetDirectoryName(c->filename);

                // make value relative to param file
                v = relativePathTo(v, paramFileDir);
                value = vislib::StringA(v).PeekBuffer();
            }
        }

        delete[] buf;

        c->pf.push_back(std::pair<std::string, std::string>(name, value));
    }

}

void utility::ParamFileManager::Load() {
    vislib::TString fn = filename;

    vislib::sys::BufferedFile pfile;
    if (!pfile.Open(fn, vislib::sys::File::READ_ONLY, vislib::sys::File::SHARE_READ, vislib::sys::File::OPEN_ONLY)) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN, "Unable to open parameter file.");
        return;
    } else {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO, "Reading parameter file \"%s\"", vislib::StringA(fn).PeekBuffer());
    }

    while (!pfile.IsEOF()) {
        vislib::StringA line = vislib::sys::ReadLineFromFileA(pfile);
        line.TrimSpaces();
        if (line.IsEmpty()) continue;
        if (line[0] == '#') continue;
        vislib::StringA::Size pos = line.Find('=');
        vislib::StringA name = line.Substring(0, pos);
        vislib::StringA value = line.Substring(pos + 1);
        value.TrimSpacesEnd();
        {
            ::megamol::console::CoreHandle hParam;
            if (::mmcGetParameterA(hCore, name, hParam, true)) {

                try {
                    // need to handle FilePathParam differently
                    // canonicalization of value is only valid for FilePathParam
                    if (!value.IsEmpty()) {
                        // resolve potential relative paths :-/
                        vislib::StringA fullPath = vislib::sys::Path::Resolve(value, vislib::StringA(vislib::sys::Path::GetDirectoryName(fn)));
                        if (vislib::sys::File::Exists(fullPath)) {
                            value = fullPath;
                        }
                    }
                } catch (vislib::sys::SystemException e) {
                    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                        "megamol::console::utility::ParamFileManager::Load: Failed to canonicalize parameter: %s\n", e.GetMsgA());
                }

                ::mmcSetParameterValueA(hParam, value);

            } else {
                vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "Unable to get handle for parameter \"%s\"\n", name.PeekBuffer());
            }
        }
    }

    pfile.Close();

}

void utility::ParamFileManager::Save() {
    ctxt c;
    c.filename = filename;
    c.useRelativeFileNames = false;
    c.hCore = hCore;
    c.comments.clear();

    ::mmcEnumParametersA(hCore, reinterpret_cast<mmcEnumStringAFunction>(&enumParameter), &c);

    vislib::sys::BufferedFile pfile;
    if (!pfile.Open(c.filename, vislib::sys::File::WRITE_ONLY, vislib::sys::File::SHARE_READ, vislib::sys::File::CREATE_OVERWRITE)) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "Unable to create parameter file.");
        return;
    } else {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO, "Writing parameter file \"%s\"", vislib::StringA(c.filename).PeekBuffer());
    }

    vislib::sys::WriteFormattedLineToFile(pfile, "#\n");
    vislib::sys::WriteFormattedLineToFile(pfile, "# Parameter file created by MegaMol™ Console\n");
    vislib::sys::WriteFormattedLineToFile(pfile, "#\n");
    vislib::sys::WriteFormattedLineToFile(pfile, "\n");

    if (!c.comments.empty()) {
        for (std::string& l : c.comments) {
            vislib::sys::WriteFormattedLineToFile(pfile, "# %s\n", l.c_str());
        }
        vislib::sys::WriteFormattedLineToFile(pfile, "\n");
    }

    for (std::pair<std::string, std::string>& p : c.pf) {
        vislib::sys::WriteFormattedLineToFile(pfile, "%s=%s\n", p.first.c_str(), p.second.c_str());
    }

    pfile.Close();

}

utility::ParamFileManager::ParamFileManager() : filename(), hCore(nullptr) {
    // intentionally empty
}

utility::ParamFileManager::~ParamFileManager() {
    hCore = nullptr; // is not owned by this class, thus is deleted elsewhere
}
