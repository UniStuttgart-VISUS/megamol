/*
 * thecam/arcball_manipulator.h
 *
 * Copyright (C) 2016 TheLib Team (http://www.thelib.org/license)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * - Neither the name of TheLib, TheLib Team, nor the names of its
 *   contributors may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THELIB TEAM AS IS AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL THELIB TEAM BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 * TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#if 0
https://www.gamedev.net/articles/programming/math-and-physics/a-simple-quaternion-based-camera-r1997/

/*
* Operator::on_mouse_down
*/
void Operator::on_mouse_down(const int button) throw() {
    POINT cursorPos;
    if (::GetCursorPos(&cursorPos)) {
        this->isLeftMouseDown = (button == 0);
        this->arcBallStart = this->MapToSphere(cursorPos.x, cursorPos.y);
    }
}


/*
* Operator::on_mouse_move
*/
void Operator::on_mouse_move(const position_type x,
    const position_type y) throw() {

    if (this->isLeftMouseDown) {
        this->arcBallCurrent = this->MapToSphere(x, y);

        auto dp = DirectX::XMVectorGetX(DirectX::XMVector3Dot(
            this->arcBallStart, this->arcBallCurrent));
        auto cp = DirectX::XMVector3Cross(this->arcBallStart,
            this->arcBallCurrent);
        auto axis = DirectX::XMVectorSet(DirectX::XMVectorGetX(cp),
            DirectX::XMVectorGetY(cp), DirectX::XMVectorGetZ(cp), 1.0f);
        auto quat = DirectX::XMQuaternionRotationNormal(axis, dp);

        auto view = DirectX::XMVectorSubtract(this->camLookAt, this->camPos);
        auto invView = DirectX::XMVectorScale(view, -1.0f);

        this->camUp = DirectX::XMVector3InverseRotate(this->camUp, quat);
        invView = DirectX::XMVector3InverseRotate(invView, quat);
        this->camPos = DirectX::XMVectorAdd(this->camLookAt, invView);

        // TODO: ultra fishy! Fix socket accessor in dispatcher?
        if (this->dispatcher.running()) {
            megamol::core::thecam::system::net::simple_message msg(
                CameraConfigurationDesc::BodySize);
            msg.set_id(CameraConfigurationDesc::ID);
            auto body = msg.body<CameraConfigurationDesc::Body>();

            megamol::core::thecam::zero_memory(body, msg.body_size());
            body->CameraPosition[0] = DirectX::XMVectorGetX(this->camPos);
            body->CameraPosition[1] = DirectX::XMVectorGetY(this->camPos);
            body->CameraPosition[2] = DirectX::XMVectorGetZ(this->camPos);
            body->FieldOfViewY = this->fovy;
            body->LookAtPosition[0] = DirectX::XMVectorGetX(this->camLookAt);
            body->LookAtPosition[1] = DirectX::XMVectorGetY(this->camLookAt);
            body->LookAtPosition[2] = DirectX::XMVectorGetZ(this->camLookAt);
            body->UpVector[0] = DirectX::XMVectorGetX(this->camUp);
            body->UpVector[1] = DirectX::XMVectorGetY(this->camUp);
            body->UpVector[2] = DirectX::XMVectorGetZ(this->camUp);

            this->dispatcher.send(msg, msg.size());
        }
    }
}

/*
* Operator::on_mouse_up
*/
void Operator::on_mouse_up(const int button) throw() {
    if (button == 0) {
        this->isLeftMouseDown = false;
    }
}

#endif


/*
 * megamol::core::thecam::arcball_manipulator<T>::arcball_manipulator
 */
template <class T>
megamol::core::thecam::arcball_manipulator<T>::arcball_manipulator(const point_type& rotCentre, const world_type radius)
    : ballRadius(radius), rotCentre(rotCentre) {}


/*
 * megamol::core::thecam::arcball_manipulator<T>::~arcball_manipulator
 */
template <class T> megamol::core::thecam::arcball_manipulator<T>::~arcball_manipulator(void) {}


/*
 * megamol::core::thecam::arcball_manipulator<T>::on_drag
 */
template <class T>
void megamol::core::thecam::arcball_manipulator<T>::on_drag(const screen_type x, const screen_type y) {
    if (this->manipulating() && this->enabled()) {
        auto cam = this->camera();
        THE_ASSERT(cam != nullptr);

        if (this->lastSx != x || this->lastSy != y) {
            this->currentVector = this->mapToSphere(x, y);

            // Compute angle and rotation quaternion.
            quaternion_type quat;
            thecam::math::set_from_vectors(quat, startVector, currentVector);

            auto const qstar = this->startRot * quat;
            auto pos = thecam::math::rotate(this->startPos - this->rotCentre, qstar * this->invStartRot) + this->rotCentre;
            cam->position(pos);
            cam->orientation(qstar);

            this->lastSx = x;
            this->lastSy = y;
        }
    }
}


/*
 * megamol::core::thecam::arcball_manipulator<T>::on_drag_start
 */
template <class T>
void megamol::core::thecam::arcball_manipulator<T>::on_drag_start(const screen_type x, const screen_type y) {
    if (!this->manipulating() && this->enabled()) {
        this->begin_manipulation();
        this->startPos = this->camera()->eye_position();
        this->invStartRot = math::invert(this->camera()->orientation());
        this->startRot = this->camera()->orientation();
        this->startVector = this->mapToSphere(x, y);
        this->lastSx = x;
        this->lastSy = y;
    }
}


/*
 * megamol::core::thecam::arcball_manipulator<T>::mapToSphere
 */
template <class T>
typename megamol::core::thecam::arcball_manipulator<T>::vector_type
megamol::core::thecam::arcball_manipulator<T>::mapToSphere(const screen_type sx, const screen_type sy) const {
    THE_ASSERT(this->camera() != nullptr);
    auto wndSize = this->camera()->resolution_gate();
    auto halfHeight = wndSize.height() / static_cast<world_type>(2);
    auto halfWidth = wndSize.width() / static_cast<world_type>(2);

    // Scale to screen
    auto bx = (sx - halfWidth) / (this->ballRadius * halfWidth);
    auto by = (sy - halfHeight) / (this->ballRadius * halfHeight);
    auto bz = static_cast<world_type>(0);

    auto mag = bx * bx + by * by;

    if (mag > 1) {
        // Point is mapped outside of the sphere: project on sphere.
        auto scale = 1.0f / std::sqrt(mag);
        bx *= scale;
        by *= scale;
    } else {
        // Point is mapped inside the sphere.
        bz = std::sqrt(1.0f - mag);
    }
    return typename maths_type::vector_type(bx, by, bz, static_cast<world_type>(0));
}
