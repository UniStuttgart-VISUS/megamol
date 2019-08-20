
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
        auto cam = this->camera();
        quaternion_type rotquat;
        auto right = cam->right_vector();
        set_from_angle_axis(rotquat, math::angle_deg2rad(angle), right);
        cam->orientation(rotquat * cam->orientation());
    }
}

/*
 * megamol::core::thecam::rotate_manipulator<T>::yaw
 */
template <class T> void megamol::core::thecam::rotate_manipulator<T>::yaw(const world_type angle) {
    if (this->enabled()) {
        auto cam = this->camera();
        quaternion_type rotquat;
        auto up = cam->up_vector();
        set_from_angle_axis(rotquat, math::angle_deg2rad(angle), up);
        cam->orientation(rotquat * cam->orientation());
    }
}

/*
 * megamol::core::thecam::rotate_manipulator<T>::roll
 */
template <class T> void megamol::core::thecam::rotate_manipulator<T>::roll(const world_type angle) {
    if (this->enabled()) {
        auto cam = this->camera();
        quaternion_type rotquat;
        auto dir = cam->view_vector();
        set_from_angle_axis(rotquat, math::angle_deg2rad(angle), dir);
        cam->orientation(rotquat * cam->orientation());
    }
}
