#pragma once

#include <string>
#include <map>
#include <optional>
#include <vector>
#include "png.h"
#include "mmcore/CoreInstance.h"

namespace megamol {
namespace core {
namespace utility {
namespace graphics {

    class MEGAMOLCORE_API ScreenShotComments {
    public:
        typedef std::map<std::string, std::string> comments_storage_map;
        typedef std::vector<png_text> png_comments;

        /**
         * Instantiate this class with a number of pre-defined comments. You can pass additional keys/values to the
         * constructor and later get out the png_text you need for feeding libpng. Note that the returned png_text array
         * is only valid as long as the ScreenShotComments instance is in scope!
         */
        ScreenShotComments(megamol::core::CoreInstance *core_instance, const std::optional<comments_storage_map> &additional_comments = std::nullopt);

        png_comments GetComments();

    private:
        comments_storage_map the_comments;
        png_comments the_vector;
    };

} // namespace graphics
} // namespace utility
} // namespace core
} // namespace megamol
