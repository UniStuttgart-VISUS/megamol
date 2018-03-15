/*
* CallCinematicCamera.cpp
*
* Copyright (C) 2017 by VISUS (Universitaet Stuttgart).
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"

#include "CallCinematicCamera.h"

#include "vislib/graphics/gl/IncludeAllGL.h"


using namespace megamol;
using namespace megamol::cinematiccamera;


/*
* CallCinematicCamera::CallCinematicCamera
*/
CallCinematicCamera::CallCinematicCamera(void) : core::AbstractGetDataCall(), 
    keyframes(NULL), boundingbox(NULL), interpolCamPos(NULL),
    selectedKeyframe(), cameraParam()
    {

    this->totalAnimTime    = 1.0f;
    this->totalSimTime     = 1.0f;
    this->dropAnimTime     = 0.0f;
    this->dropSimTime      = 0.0f;
    this->interpolSteps    = 10;
    this->fps              = 24;
    this->bboxCenter       = vislib::math::Point<float, 3>(0.0f, 0.0f, 0.0f);
    this->firstCtrllPos    = vislib::math::Vector<float, 3>(0.0f, 0.0f, 0.0f);
    this->lastCtrllPos     = vislib::math::Vector<float, 3>(0.0f, 0.0f, 0.0f);
}


/*
* CallCinematicCamera::~CallCinematicCamera
*/
CallCinematicCamera::~CallCinematicCamera(void) {
	this->keyframes      = NULL;
    this->interpolCamPos = NULL;
    this->boundingbox    = NULL;
}


/*
* CallCinematicCamera::oglCheckErrors
*/
void CallCinematicCamera::OGLCheckErrors(const char *file, const char *func, int line) const {
#if DEBUG 

    GLenum err;
    vislib::StringA errString;

    vislib::StringA fileString(file);
    int index = fileString.FindLast("\\"); // WIN
    if (index < 0) {
        index = fileString.FindLast("/"); // LINUX
    }
    if (index > 0) {
        fileString = fileString.Substring(index + 1);
    }

    while ((err = glGetError()) != GL_NO_ERROR) {

        switch (err) {
        case(GL_INVALID_ENUM):                  errString = "GL_INVALID_ENUM";                  break;
            //An unacceptable value is specified for an enumerated argument. The offending command is ignored and has no other side effect than to set the error flag.
        case(GL_INVALID_VALUE):                 errString = "GL_INVALID_VALUE";                 break;
            // A numeric argument is out of range. The offending command is ignored and has no other side effect than to set the error flag.
        case(GL_INVALID_OPERATION):             errString = "GL_INVALID_OPERATION";             break;
            // The specified operation is not allowed in the current state. The offending command is ignored and has no other side effect than to set the error flag.
        case(GL_INVALID_FRAMEBUFFER_OPERATION): errString = "GL_INVALID_FRAMEBUFFER_OPERATION"; break;
            // The command is trying to render to or read from the framebuffer while the currently bound framebuffer is not framebuffer complete(i.e.the return value from glCheckFramebufferStatus is not GL_FRAMEBUFFER_COMPLETE). The offending command is ignored and has no other side effect than to set the error flag.
        case(GL_OUT_OF_MEMORY):                 errString = "GL_OUT_OF_MEMORY";                 break;
            // There is not enough memory left to execute the command.The state of the GL is undefined, except for the state of the error flags, after this error is recorded.
        case(GL_STACK_UNDERFLOW):               errString = "GL_OUT_OF_MEMORY";                 break;
            // An attempt has been made to perform an operation that would cause an internal stack to underflow.
        case(GL_STACK_OVERFLOW):                errString = "GL_OUT_OF_MEMORY";                 break;
            // An attempt has been made to perform an operation that would cause an internal stack to overflow. 
        default:                                errString = "___UNKNOWN_ERROR___";                    break;
        }

        vislib::sys::Log::DefaultLog.WriteError(">>> OPENGL ERROR | FILE= %s | FUNCTION= %s | LINE= %i | GL_ERROR= %s <<<\n", fileString.PeekBuffer(), func, line, errString.PeekBuffer());
    }

#endif
}