#include <tuple>

#include "SampleSphere.h"

#include "mmcore/BoundingBoxes_2.h"

namespace megamol::core::utility {
std::tuple<std::vector<glm::vec3>, std::vector<glm::vec3>> orbital_camera_samples(
    core::BoundingBoxes_2 const& bboxes, unsigned int num_samples) {
    auto const& obb = bboxes.BoundingBox();
    auto const longest_edge = obb.LongestEdge();
    auto const center_vl = obb.CalcCenter(); // look-at
    auto const center = glm::vec3(center_vl.GetX(), center_vl.GetY(), center_vl.GetZ());
    auto max_dist = 0.6f * longest_edge;

    auto cam_positions = sample_on_sphere(max_dist, num_samples, 42); // actually also directions before adding center
    auto cam_directions = cam_positions;
    std::transform(cam_positions.begin(), cam_positions.end(), cam_positions.begin(),
        [&center](auto const& pos) { return pos + center; });
    return std::make_tuple(cam_positions, cam_directions);
}
} // namespace megamol::core::utility
