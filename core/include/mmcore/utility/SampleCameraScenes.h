#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <glm/glm.hpp>

#include "mmcore/view/AbstractViewInterface.h"

namespace megamol::core::utility {
using cam_samples_func = std::function<std::tuple<std::vector<glm::vec3>, std::vector<glm::vec3>>(
    megamol::core::BoundingBoxes_2, unsigned int)>;

cam_samples_func GetCamScenesFunctional(std::string camera_path_pattern);

std::string SampleCameraScenes(std::shared_ptr<megamol::core::view::AbstractViewInterface> view,
    cam_samples_func cam_func, unsigned int num_samples);
} // namespace megamol::core::utility
