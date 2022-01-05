/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */
#include "mmcore/utility/platform/RuntimeInfo.h"

#include <array>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <vector>

#ifdef _WIN32
#include "mmcore/utility/platform/WMIUtil.h"
#include <tchar.h>
#include <tlhelp32.h>
#include <windows.h>
#define the_popen _popen
#define the_pclose _pclose
#else
#include <link.h>
#define the_popen popen
#define the_pclose pclose
#endif

namespace {

#ifdef _WIN32

std::string get_file_version(const char* path) {
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
                        ret += std::to_string((verInfo->dwFileVersionMS >> 16) & 0xffff) + "." +
                               std::to_string((verInfo->dwFileVersionMS >> 0) & 0xffff) + "." +
                               std::to_string((verInfo->dwFileVersionLS >> 16) & 0xffff) + "." +
                               std::to_string((verInfo->dwFileVersionLS >> 0) & 0xffff);
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
[[maybe_unused]] std::vector<std::string> dlinfo_search_path(void* handle) {
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
    for (unsigned int i = 0; i < serinfo.dls_cnt; i++) {
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

void megamol::core::utility::platform::RuntimeInfo::get_hardware_info() {
#ifdef _WIN32
    //m_hardware_info = execute("systeminfo");
    std::stringstream s;
    s << "{" << std::endl;
    s << R"("Processor Name":")" << wmi.get_value("Win32_Processor", "Name") << "\"," << std::endl;
    s << R"("Processor Version":")" << wmi.get_value("Win32_Processor", "Version") << "\"," << std::endl;
    s << R"("GPU Name":")" << wmi.get_value("Win32_VideoController", "Name") << "\"," << std::endl;
    s << R"("OS Name":")" << wmi.get_value("Win32_OperatingSystem", "Name") << "\"," << std::endl;
    s << R"("OS Version":")" << wmi.get_value("Win32_OperatingSystem", "Version") << "\"," << std::endl;
    s << R"("OS Architecture":")" << wmi.get_value("Win32_OperatingSystem", "OSArchitecture") << "\"," << std::endl;
    s << R"("Available Memory":")" << wmi.get_value("Win32_OperatingSystem", "TotalVisibleMemorySize") << "\""
      << std::endl;
    s << "}";
    m_hardware_info = s.str();
#else
    m_hardware_info = execute("cat /proc/cpuinfo /proc/meminfo");
#endif
}

void megamol::core::utility::platform::RuntimeInfo::get_runtime_libraries() {
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
                out << get_file_version(me32.szExePath) << ")" << std::endl;
            } while (Module32Next(h_mod_snap, &me32));
        }
        CloseHandle(h_mod_snap);
        m_runtime_libraries = out.str();
    } else {
        m_runtime_libraries = "";
    }
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
        out << std::endl;
    }
    m_runtime_libraries = out.str();
#endif
}

void megamol::core::utility::platform::RuntimeInfo::get_os_info() {
#ifdef _WIN32
    m_os_info = execute("ver");
#else
    m_os_info = execute("cat /etc/issue");
#endif
}


std::string megamol::core::utility::platform::RuntimeInfo::execute(const std::string& cmd) {
    std::array<char, 1024> buffer;
    std::string result;

    auto pipe = the_popen(cmd.c_str(), "r");

    if (!pipe)
        throw std::runtime_error("popen() failed!");

    while (!feof(pipe)) {
        if (fgets(buffer.data(), buffer.size(), pipe) != nullptr)
            result += buffer.data();
    }

    auto rc = the_pclose(pipe);

    if (rc == EXIT_SUCCESS) {
        return result;
    } else {
        return "unable to execute " + cmd;
    }
}
