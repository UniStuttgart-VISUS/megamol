#include "mmcore/utility/SampleCameraScenes.h"

#include "mmcore/BoundingBoxes_2.h"
#include "mmcore/utility/LongestEdgeCameraSamples.h"
#include "mmcore/utility/OrbitalCameraSamples.h"
#include "mmcore/view/CameraSerializer.h"


megamol::core::utility::cam_samples_func megamol::core::utility::GetCamScenesFunctional(
    std::string camera_path_pattern) {
    std::function<std::tuple<std::vector<glm::vec3>, std::vector<glm::vec3>>(
        megamol::core::BoundingBoxes_2, unsigned int)>
        sampler;
    if (camera_path_pattern == "orbit") {
        return &megamol::core::utility::orbital_camera_samples;
    } else if (camera_path_pattern == "longest_edge") {
        return &megamol::core::utility::longest_edge_camera_samples;
    }
    return cam_samples_func();
}


std::string megamol::core::utility::SampleCameraScenes(std::shared_ptr<megamol::core::view::AbstractViewInterface> view,
    cam_samples_func cam_func, unsigned int num_samples) {
    auto [cam_positions, cam_directions] = cam_func(view->GetBoundingBoxes(), num_samples);

    auto cam = view->GetCamera();

    auto const base_pose = cam.getPose();

    std::vector<megamol::core::view::Camera> cameras(cam_positions.size());

    std::transform(cam_positions.begin(), cam_positions.end(), cam_directions.begin(), cameras.begin(),
        [&base_pose, &cam](auto const& pos, auto const& dir) {
            auto pose = base_pose;
            pose.position = pos;
            pose.direction = dir;
            pose.up = glm::vec3(0, 1, 0);
            pose.right = glm::normalize(glm::cross(pose.up, dir));
            pose.up = glm::normalize(glm::cross(pose.right, dir));
            cam.setPose(pose);
            return cam;
        });

    auto serializer = megamol::core::view::CameraSerializer();

    return serializer.serialize(cameras);
}
