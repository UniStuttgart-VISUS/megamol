/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */
#include "mmcore/utility/platform/RuntimeInfo.h"

#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <tlhelp32.h>
#include <tchar.h>
#else
#include <link.h>
#endif

namespace {

#ifdef _WIN32

std::string getFileVersion(const char* path) {
    // https://stackoverflow.com/questions/940707/how-do-i-programmatically-get-the-version-of-a-dll-or-exe-file
    std::string ret;
    DWORD verHandle = 0;
    UINT size = 0;
    LPBYTE lpBuffer = NULL;
    DWORD verSize = GetFileVersionInfoSize(path, &verHandle);

    if (verSize != NULL) {
        LPSTR verData = new char[verSize];

        if (GetFileVersionInfo(path, verHandle, verSize, verData)) {
            if (VerQueryValue(verData, "\\", reinterpret_cast<void**>(&lpBuffer), &size)) {
                if (size) {
                    VS_FIXEDFILEINFO* verInfo = reinterpret_cast<VS_FIXEDFILEINFO*>(lpBuffer);
                    if (verInfo->dwSignature == 0xfeef04bd) {

                        // Doesn't matter if you are on 32 bit or 64 bit,
                        // DWORD is always 32 bits, so first two revision numbers
                        // come from dwFileVersionMS, last two come from dwFileVersionLS
                        ret +=
                            std::to_string((verInfo->dwFileVersionMS >> 16) & 0xffff)
                            + "." + std::to_string((verInfo->dwFileVersionMS >> 0) & 0xffff)
                            + "." + std::to_string((verInfo->dwFileVersionLS >> 16) & 0xffff)
                            + "." + std::to_string((verInfo->dwFileVersionLS >> 0) & 0xffff);
                    }
                }
            }
        }
        delete[] verData;
    }
    return ret;
}

#else

/**
 * Extract list of library search paths from a dlopen(nullptr, RTLD_NOW) handle.
 * @param handle A handle generated with dlopen(nullptr, RTLD_NOW).
 * @return List of library search paths.
 */
std::vector<std::string> dlinfo_search_path(void* handle) {
    // `man dlinfo`
    std::vector<std::string> paths;
    Dl_serinfo serinfo;
    if (dlinfo(handle, RTLD_DI_SERINFOSIZE, &serinfo) != 0) {
        throw std::runtime_error(std::string("Error from dlinfo(): ") + dlerror());
    }
    auto* sip = reinterpret_cast<Dl_serinfo*>(std::malloc(serinfo.dls_size));
    if (dlinfo(handle, RTLD_DI_SERINFOSIZE, sip) != 0) {
        std::free(sip);
        throw std::runtime_error(std::string("Error from dlinfo(): ") + dlerror());
    }
    if (dlinfo(handle, RTLD_DI_SERINFO, sip) != 0) {
        std::free(sip);
        throw std::runtime_error(std::string("Error from dlinfo(): ") + dlerror());
    }
    paths.resize(serinfo.dls_cnt);
    for (int i = 0; i < serinfo.dls_cnt; i++) {
        paths[i] = std::string(sip->dls_serpath[i].dls_name);
    }
    std::free(sip);
    return paths;
}

/**
 * Extract list of loaded libraries from a dlopen(nullptr, RTLD_NOW) handle.
 * @param handle A handle generated with dlopen(nullptr, RTLD_NOW).
 * @return List of loaded libraries.
 */
std::vector<std::string> dlinfo_linkmap(void* handle) {
    std::vector<std::string> list;
    struct link_map* map = nullptr;
    if (dlinfo(handle, RTLD_DI_LINKMAP, &map) != 0) {
        throw std::runtime_error(std::string("Error from dlinfo(): ") + dlerror());
    }
    map = map->l_next; // ignore current exe itself
    while (map != nullptr) {
        list.emplace_back(std::string(map->l_name));
        map = map->l_next;
    }
    return list;
}

#endif

} // namespace

std::string megamol::core::utility::platform::getRuntimeLibraries() {
#ifdef _WIN32
    HANDLE h_mod_snap = INVALID_HANDLE_VALUE;
    h_mod_snap = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, GetCurrentProcessId());
    if (h_mod_snap != INVALID_HANDLE_VALUE) {
        std::stringstream out;
        MODULEENTRY32 me32;
        me32.dwSize = sizeof(MODULEENTRY32);
        if (Module32First(h_mod_snap, &me32)) {
            do {
                out << me32.szExePath << " (";
                out << getFileVersion(me32.szExePath) << ");";
            } while (Module32Next(h_mod_snap, &me32));
        }
        CloseHandle(h_mod_snap);
        return out.str();
    }
    return "";
#else
    void* handle = dlopen(nullptr, RTLD_NOW);

    // TODO looks like all library paths are already absolute, do we need search paths here?
    // const auto paths = dlinfo_search_path(handle);

    const auto list = dlinfo_linkmap(handle);

    std::stringstream out;
    for (const auto& lib : list) {
        out << lib;
        // If the library is a symlink, print link target to get the filename with the full version number.
        std::filesystem::path p(lib);
        if (std::filesystem::is_symlink(p)) {
            p = std::filesystem::canonical(p);
            out << " (=> " << p.string() << ")";
        }
        out << ";";
    }
    return out.str();
#endif
}
