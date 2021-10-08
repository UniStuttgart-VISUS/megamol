#include "mmcore/utility/graphics/ScreenShotComments.h"

#include "megamol_build_info.h"
#include "mmcore/utility/DateTime.h"

#ifdef _WIN32
#include <windows.h>
#include <tlhelp32.h>
#include <tchar.h>
#else
#include <link.h>
#endif

namespace mcu_graphics = megamol::core::utility::graphics;

mcu_graphics::ScreenShotComments::ScreenShotComments(std::string const& project_configuration, const std::optional<comments_storage_map> &additional_comments) {

    if (m_moduleInfo.empty()) {
        init_moduleInfo();
    }

    the_comments["Title"] = "MegaMol Screen Capture " + utility::DateTime::CurrentDateTimeFormatted();
    //the_comments["Author"] = "";
    //the_comments["Description"] = "";
    the_comments["MegaMol project"] = project_configuration;
    //the_comments["Copyright"] = "";
    the_comments["Creation Time"] = utility::DateTime::CurrentDateTimeFormatted();
    the_comments["Software"] = "MegaMol " + std::string(megamol::build_info::MEGAMOL_VERSION) + "-" + std::string(megamol::build_info::MEGAMOL_GIT_HASH);
    the_comments["CMakeCache"] = std::string(megamol::build_info::MEGAMOL_CMAKE_CACHE);
    the_comments["RemoteBranch"] = megamol::build_info::MEGAMOL_GIT_BRANCH_NAME_FULL;
    the_comments["RemoteURL"] = megamol::build_info::MEGAMOL_GIT_REMOTE_URL;
    the_comments["Environment"] = m_moduleInfo;

    //the_comments["Disclaimer"] = "";
    //the_comments["Warning"] = "";
    //the_comments["Source"] = "";
    //the_comments["Comment"] = "";

    if (additional_comments.has_value()) {
        // add/overwrite default comments
        for (auto& k : additional_comments.value()) {
            the_comments[k.first] = k.second;
        }
    }

    for (auto& s : the_comments) {
        the_vector.emplace_back();
        the_vector.back().compression = PNG_TEXT_COMPRESSION_NONE;
        // what are the libpng people thinking?
        the_vector.back().key = const_cast<png_charp>(static_cast<png_const_charp>(s.first.data()));
        the_vector.back().text = static_cast<png_charp>(s.second.data());
        the_vector.back().text_length = s.second.size();
    }
}

mcu_graphics::ScreenShotComments::png_comments mcu_graphics::ScreenShotComments::GetComments() const {
    return the_vector;
}


std::string mcu_graphics::ScreenShotComments::GetProjectFromPNG(const std::filesystem::path filename) {
    std::string content;
    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("ScreenShotComments::GetProjectFromPNG: Unable to create png struct");
    } else {
#ifdef _MSC_VER
        FILE* fp = _wfopen(filename.native().c_str(), L"rb");
#else
        FILE* fp = fopen(filename.native().c_str(), "rb");
#endif
        if (fp == nullptr) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "ScreenShotComments::GetProjectFromPNG: Unable to open png file \"%s\"", filename.generic_u8string().c_str());
        } else {
            png_infop info = png_create_info_struct(png);
            if (!info) {
                megamol::core::utility::log::Log::DefaultLog.WriteError("ScreenShotComments::GetProjectFromPNG: Unable to create png info struct");
            } else {
                setjmp(png_jmpbuf(png));
                png_init_io(png, fp);
                png_read_info(png, info);

                png_textp texts;
                int num_text = 0;
                png_get_text(png, info, &texts, &num_text);
                bool found = false;
                for (int i = 0; i < num_text; ++i) {
                    if (strcmp(texts[i].key, "MegaMol project") == 0) {
                        found = true;
                        content = std::string(texts[i].text);
                    }
                }

                if (!found) {
                    png_uint_32 exif_size = 0;
                    png_bytep exif_data = nullptr;
                    png_get_eXIf_1(png, info, &exif_size, &exif_data);
                    if (exif_size > 0) {
                        found = true;
                        content = reinterpret_cast<char*>(exif_data);
                    }
                }
                if (!found) {
                    megamol::core::utility::log::Log::DefaultLog.WriteError("LoadProject: Unable to extract png text or exif data");
                }
                png_destroy_info_struct(png, &info);
            }
            fclose(fp);
        }
        png_destroy_read_struct(&png, nullptr, nullptr);
        // exif_data buffer seems to live inside exif_info and is disposed automatically
    }
    return content;
}

bool megamol::core::utility::graphics::ScreenShotComments::EndsWith(
    const std::string& filename, const std::string& suffix) {
    if (suffix.size() > filename.size()) return false;
    return std::equal(suffix.rbegin(), suffix.rend(), filename.rbegin());
}

bool megamol::core::utility::graphics::ScreenShotComments::EndsWithCaseInsensitive(
    const std::string& filename, const std::string& suffix) {
    if (suffix.size() > filename.size()) return false;
    return std::equal(suffix.rbegin(), suffix.rend(), filename.rbegin(),
        [](const char a, const char b) { return tolower(a) == tolower(b); });
}

void megamol::core::utility::graphics::ScreenShotComments::init_moduleInfo() {
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
                out << getFileVersion(me32.szExePath) << ");";
            } while (Module32Next(h_mod_snap, &me32));
        }
        CloseHandle(h_mod_snap);
        m_moduleInfo = out.str();
    }
#else
    struct link_map* map = reinterpret_cast<struct link_map*>(dlopen(NULL, RTLD_NOW));
    map = map->l_next;
    std::stringstream out;
    while (map) {
        // TODO what about the version information here
        out << map->l_name << ";";
        map = map->l_next;
    }
    m_moduleInfo = out.str();
#endif
}

std::string megamol::core::utility::graphics::ScreenShotComments::getFileVersion(const char* path) {
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
