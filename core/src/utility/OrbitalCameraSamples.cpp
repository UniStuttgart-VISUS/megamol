#include "mmcore/utility/OrbitalCameraSamples.h"

std::tuple<std::vector<glm::vec3>, std::vector<glm::vec3>> megamol::core::utility::orbital_camera_samples(
    core::BoundingBoxes_2 const& bboxes, unsigned int num_samples) {
    auto const& obb = bboxes.BoundingBox();
    auto const center_vl = obb.CalcCenter(); // look-at
    auto const center = glm::vec3(center_vl.GetX(), center_vl.GetY(), center_vl.GetZ());
    const auto max_dist =
        sqrtf(obb.Width() * obb.Width() + obb.Height() * obb.Height() + obb.Depth() * obb.Depth());

    auto cam_positions = sample_on_sphere(max_dist, num_samples, 42);
    auto cam_directions = cam_positions;
    std::transform(cam_positions.begin(), cam_positions.end(), cam_positions.begin(),
        [&center](auto const& pos) { return pos + center; });
    std::transform(cam_directions.begin(), cam_directions.end(), cam_directions.begin(),
        [](auto const& dir) { return glm::normalize(-dir); });
    return std::make_tuple(cam_positions, cam_directions);
}
