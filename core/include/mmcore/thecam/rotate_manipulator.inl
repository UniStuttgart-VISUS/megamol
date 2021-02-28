
/*
 * megamol::core::thecam::rotate_manipulator<T>::rotate_manipulator
 */
template <class T>
megamol::core::thecam::rotate_manipulator<T>::rotate_manipulator(const world_type angle) : rotationAngle(angle) {}

/*
 * megamol::core::thecam::rotate_manipulator<T>::~rotate_manipulator
 */
template <class T> megamol::core::thecam::rotate_manipulator<T>::~rotate_manipulator(void) {}

/*
 * megamol::core::thecam::rotate_manipulator<T>::pitch
 */
template <class T> void megamol::core::thecam::rotate_manipulator<T>::pitch(const world_type angle) {
    if (this->enabled()) {
        auto cam_pose = this->camera()->get<megamol::core::view::Camera::Pose>();
        auto right = glm::cross(cam_pose.direction, cam_pose.up);
        cam_pose.direction = glm::rotate(cam_pose.direction,math::angle_deg2rad(angle), right);
        cam_pose.up = glm::rotate(cam_pose.up, math::angle_deg2rad(angle), right);
        this->camera()->setPose(cam_pose);
    }
}

/*
 * megamol::core::thecam::rotate_manipulator<T>::yaw
 */
template <class T> void megamol::core::thecam::rotate_manipulator<T>::yaw(const world_type angle, bool fixToWorldUp) {
    if (this->enabled()) {
        auto cam_pose = this->camera()->get<core::view::Camera::Pose>();
        auto up = fixToWorldUp ? vector_type(0.0f, 1.0f, 0.0f, 0.0f) : cam_pose.up;
        cam_pose.direction = glm::rotate(cam_pose.direction, math::angle_deg2rad(angle), up);
        cam_pose.up = glm::rotate(cam_pose.up, math::angle_deg2rad(angle), up);
        this->camera()->setPose(cam_pose);
    }
}

/*
 * megamol::core::thecam::rotate_manipulator<T>::roll
 */
template <class T> void megamol::core::thecam::rotate_manipulator<T>::roll(const world_type angle) {
    if (this->enabled()) {
        auto cam_pose = this->camera()->get<core::view::Camera::Pose>();
        cam_pose.up = glm::rotate(cam_pose.up, math::angle_deg2rad(angle), cam_pose.direction);
        this->camera()->setPose(cam_pose);
    }
}
