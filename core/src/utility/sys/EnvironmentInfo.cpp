#include "mmcore/utility/sys/EnvironmentInfo.h"

#include <sstream>
#ifdef _WIN32
#include <windows.h>
#include <tlhelp32.h>
#include <tchar.h>
#else
#include <link.h>
#endif

void megamol::core::utility::sys::EnvironmentInfo::init_module_info() {
#ifdef WIN32
    HANDLE h_mod_snap = INVALID_HANDLE_VALUE;
    h_mod_snap = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE,GetCurrentProcessId());
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
        m_module_info = out.str();
    }
#else
    struct link_map* map = reinterpret_cast<struct link_map*>(dlopen(NULL, RTLD_NOW));
    map = map->l_next;
    std::stringstream out;
    while (map) {
        // TODO what about the version information here
        out << map->l_name << ";" << std::endl;
        map = map->l_next;
    }
    m_moduleInfo = out.str();
#endif
}

std::string megamol::core::utility::sys::EnvironmentInfo::get_file_version(const char* path) {
    std::string ret;
#ifdef WIN32
    // https://stackoverflow.com/questions/940707/how-do-i-programmatically-get-the-version-of-a-dll-or-exe-file
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
#endif
    return ret;
}
