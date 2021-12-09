#include "mmcore/utility/graphics/ScreenShotComments.h"

#include "megamol_build_info.h"
#include "mmcore/utility/DateTime.h"
#include "mmcore/utility/platform/RuntimeInfo.h"
#include "cmrc/cmrc.hpp"

#ifdef _WIN32
#include <tchar.h>
#include <windows.h>
#endif

CMRC_DECLARE(megamolrc);

namespace mcu_graphics = megamol::core::utility::graphics;

mcu_graphics::ScreenShotComments::ScreenShotComments(
    std::string const& project_configuration, const std::optional<comments_storage_map>& additional_comments) {

    auto rs_fs = cmrc::megamolrc::get_filesystem();
    auto the_cache = rs_fs.open("CMakeCache.txt");
    auto the_diff = rs_fs.open("GitDiff.txt");

    the_comments["Title"] = "MegaMol Screen Capture " + utility::DateTime::CurrentDateTimeFormatted();
    //the_comments["Author"] = "";
    //the_comments["Description"] = "";
    //the_comments["Copyright"] = "";
    the_comments["Creation Time"] = utility::DateTime::CurrentDateTimeFormatted();
    the_comments["Software"] = "MegaMol " + std::string(megamol::build_info::MEGAMOL_VERSION) + "-" +
                               std::string(megamol::build_info::MEGAMOL_GIT_HASH);
    the_comments["MegaMol project"] = project_configuration;

    the_comments["Remote Branch"] = megamol::build_info::MEGAMOL_GIT_BRANCH_NAME_FULL;
    the_comments["Remote URL"] = megamol::build_info::MEGAMOL_GIT_REMOTE_URL;
    the_comments["Software Environment"] = platform::RuntimeInfo::GetRuntimeLibraries();
    the_comments["Hardware Environment"] = platform::RuntimeInfo::GetHardwareInfo();
    //the_comments["CMakeCache"] = std::string(megamol::build_info::MEGAMOL_CMAKE_CACHE);
    the_comments["CMakeCache"] = std::string(the_cache.begin(), the_cache.end());
    the_comments["Git Diff"] = std::string(the_diff.begin(), the_diff.end());
    the_comments["Operating System"] = platform::RuntimeInfo::GetOsInfo();

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
        if (s.second.size() > 1024) {
            the_vector.back().compression = PNG_TEXT_COMPRESSION_zTXt;
        } else {
            the_vector.back().compression = PNG_TEXT_COMPRESSION_NONE;
        }
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
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "ScreenShotComments::GetProjectFromPNG: Unable to create png struct");
    } else {
#ifdef _MSC_VER
        FILE* fp = _wfopen(filename.native().c_str(), L"rb");
#else
        FILE* fp = fopen(filename.native().c_str(), "rb");
#endif
        if (fp == nullptr) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "ScreenShotComments::GetProjectFromPNG: Unable to open png file \"%s\"",
                filename.generic_u8string().c_str());
        } else {
            png_infop info = png_create_info_struct(png);
            if (!info) {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "ScreenShotComments::GetProjectFromPNG: Unable to create png info struct");
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
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "LoadProject: Unable to extract png text or exif data");
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
    if (suffix.size() > filename.size())
        return false;
    return std::equal(suffix.rbegin(), suffix.rend(), filename.rbegin());
}

bool megamol::core::utility::graphics::ScreenShotComments::EndsWithCaseInsensitive(
    const std::string& filename, const std::string& suffix) {
    if (suffix.size() > filename.size())
        return false;
    return std::equal(suffix.rbegin(), suffix.rend(), filename.rbegin(),
        [](const char a, const char b) { return tolower(a) == tolower(b); });
}
