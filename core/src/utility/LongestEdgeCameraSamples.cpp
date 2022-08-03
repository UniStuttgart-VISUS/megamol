#include "mmcore/utility/LongestEdgeCameraSamples.h"

std::tuple<std::vector<glm::vec3>, std::vector<glm::vec3>> megamol::core::utility::longest_edge_camera_samples(
    core::BoundingBoxes_2 const& bboxes, unsigned int num_samples) {
    auto const& obb = bboxes.BoundingBox();
    auto const center_vl = obb.CalcCenter();
    auto const center = glm::vec3(center_vl.GetX(), center_vl.GetY(), center_vl.GetZ());

    auto const longest_v = glm::vec3(obb.Width(), obb.Height(), obb.Depth());
    auto idx = 0u;
    auto longest = longest_v.x;
    glm::vec3 cam_dir = glm::vec3(1, 0, 0);
    if (longest_v.y > longest) {
        idx = 1;
        longest = longest_v.y;
        cam_dir = glm::vec3(0, 1, 0);
    }
    if (longest_v.z > longest) {
        idx = 2;
        longest = longest_v.z;
        cam_dir = glm::vec3(0, 0, 1);
    }

    std::vector<glm::vec3> cam_positions(num_samples);
    std::vector<glm::vec3> cam_directions(num_samples, cam_dir);

    auto start = center - cam_dir * longest;
    auto incr = 1.5f * longest / num_samples;

    for (unsigned int i = 0; i < num_samples; ++i) {
        cam_positions[i] = start + cam_dir * (incr * static_cast<float>(i));
    }

    return std::make_tuple(cam_positions, cam_directions);
}
