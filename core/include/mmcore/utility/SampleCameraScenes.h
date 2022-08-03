#include <memory>
#include <string>

#include "mmcore/view/AbstractViewInterface.h"

namespace megamol::core::utility {
std::string SampleCameraScenes(std::shared_ptr<megamol::core::view::AbstractViewInterface> view,
    std::string camera_path_pattern, unsigned int num_samples);
}
