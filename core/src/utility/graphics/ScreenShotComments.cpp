#include "mmcore/utility/graphics/ScreenShotComments.h"

#include "mmcore/versioninfo.h"
#include "mmcore/utility/DateTime.h"

namespace mcu_graphics = megamol::core::utility::graphics;

mcu_graphics::ScreenShotComments::ScreenShotComments(megamol::core::CoreInstance *core_instance, const std::optional<comments_storage_map> &additional_comments) {

    std::string serInstances, serModules, serCalls, serParams;
    core_instance->SerializeGraph(serInstances, serModules, serCalls, serParams);
    const auto config_string = serInstances + "\n" + serModules + "\n" + serCalls + "\n" + serParams;

    the_comments["Title"] = "MegaMol Screen Shot " + utility::DateTime::CurrentDateTimeFormatted();
    //the_comments["Author"] = "";
    //the_comments["Description"] = "";
    the_comments["MegaMol project"] = config_string;
    //the_comments["Copyright"] = "";
    the_comments["Creation Time"] = utility::DateTime::CurrentDateTimeFormatted();
    the_comments["Software"] = "MegaMol " + std::to_string(megamol::core::MEGAMOL_VERSION_MAJOR) + "." + std::to_string(MEGAMOL_CORE_MINOR_VER) + "." + MEGAMOL_CORE_COMP_REV;
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

mcu_graphics::ScreenShotComments::png_comments mcu_graphics::ScreenShotComments::GetComments() {
    return the_vector;
}
