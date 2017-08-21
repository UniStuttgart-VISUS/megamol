/*
* KeyframeKeeper.cpp
*
*/


#include "stdafx.h"

#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/AbstractSlot.h"
#include "mmcore/utility/xml/XmlParser.h"
#include "mmcore/utility/xml/XmlReader.h"

#include "vislib/math/Point.h"
#include "vislib/math/Vector.h"
#include "vislib/StringSerialiser.h"
#include "vislib/assert.h"

#include <iostream>
#include <fstream>

#include "KeyframeKeeper.h"
#include "CallCinematicCamera.h"


using namespace megamol;
using namespace megamol::cinematiccamera;
using namespace vislib;
using namespace vislib::math;
using namespace megamol::core;


/*
* KeyframeKeeper::KeyframeKeeper
*/
KeyframeKeeper::KeyframeKeeper(void) : core::Module(),
    cinematicCallSlot("scene3D", "holds keyframe data"),
    addKeyframeParam(             "01 Add keyframe", "Adds new eyframe at the currently selected position inbetween two keyframes"),
    timeForNewKeyframeParam(      "02 Time for new keyframe", "Time of new keyframe."),
    deleteSelectedKeyframeParam(  "03 Delete keyframe", "Deletes the currently selected keyframe."),
    editCurrentTimeParam(         "04 Edit TIME of selected keyframe", "Edit time of the selected keyframe."),
    editCurrentPosParam(          "05 Edit POSITION of selected keyframe", "Edit  position vector of the selected keyframe."),
    editCurrentLookAtParam(       "06 Edit LOOKAT of selected keyframe", "Edit LookAt vector of the selected keyframe."),
    editCurrentUpParam(           "07 Edit UP of selected keyframe", "Edit Up vector of the selected keyframe."),
    setTotalTimeParam(            "08 Total time", "The total timespan of the movie in seconds."),
    fileNameParam(                "09 Filename", "The name of the file to load or save."),
    saveKeyframesParam(           "10 Save Keyframes", "Saves keyframes to file."),
    loadKeyframesParam(           "11 Load Keyframes", "Loads keyframes from file."),
    autoLoadKeyframesAtStartParam("12 Auto load keyframes at start", "Automatically load keyframes from file at start."),
    selectedKeyframe(), cameraParam(NULL)
    {

	// setting up callback
	this->cinematicCallSlot.SetCallback(CallCinematicCamera::ClassName(),
		CallCinematicCamera::FunctionName(CallCinematicCamera::CallForUpdateKeyframeKeeper), &KeyframeKeeper::CallForUpdateKeyframeKeeper);
	this->MakeSlotAvailable(&this->cinematicCallSlot);


    // init variables
    this->keyframes.Clear();
    this->keyframes.AssertCapacity(1000);
    this->boundingBox.SetNull();
    this->totalTime            = 1.0f;
    this->filename             = ".\\keyframes.kf";

    // init parameters

    this->addKeyframeParam.SetParameter(new param::ButtonParam('a'));
    this->MakeSlotAvailable(&this->addKeyframeParam);

    this->timeForNewKeyframeParam.SetParameter(new param::FloatParam(0.0f, 0.0f));
    this->MakeSlotAvailable(&this->timeForNewKeyframeParam);

    this->deleteSelectedKeyframeParam.SetParameter(new param::ButtonParam('d'));
    this->MakeSlotAvailable(&this->deleteSelectedKeyframeParam);

    this->editCurrentTimeParam.SetParameter(new param::FloatParam(this->selectedKeyframe.getTime()));
    this->MakeSlotAvailable(&this->editCurrentTimeParam);

    this->editCurrentPosParam.SetParameter(new param::Vector3fParam(this->selectedKeyframe.getCamPosition()));
    this->MakeSlotAvailable(&this->editCurrentPosParam);

    this->editCurrentLookAtParam.SetParameter(new param::Vector3fParam(this->selectedKeyframe.getCamLookAt()));
    this->MakeSlotAvailable(&this->editCurrentLookAtParam);

    this->editCurrentUpParam.SetParameter(new param::Vector3fParam(this->selectedKeyframe.getCamUp()));
    this->MakeSlotAvailable(&this->editCurrentUpParam);

    this->setTotalTimeParam.SetParameter(new param::FloatParam(1.0f, 0.0f));
    this->MakeSlotAvailable(&this->setTotalTimeParam);

    this->fileNameParam.SetParameter(new param::FilePathParam(this->filename));
    this->MakeSlotAvailable(&this->fileNameParam);

	this->saveKeyframesParam.SetParameter(new param::ButtonParam());
	this->MakeSlotAvailable(&this->saveKeyframesParam);

	this->loadKeyframesParam.SetParameter(new param::ButtonParam());
	this->MakeSlotAvailable(&this->loadKeyframesParam);

    this->autoLoadKeyframesAtStartParam.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->autoLoadKeyframesAtStartParam);
}


/*
* KeyframeKeeper::~KeyframeKeeper
*/
KeyframeKeeper::~KeyframeKeeper(void) {

	this->Release();
}


/*
* KeyframeKeeper::create(void)
*/
bool KeyframeKeeper::create(void) {

	return true;
}


/*
* KeyframeKeeper::release(void)
*/
void KeyframeKeeper::release(void) {

	// intentionally empty
}


/*
* KeyframeKeeper::CallForUpdateKeyframeKeeper
*/
bool KeyframeKeeper::CallForUpdateKeyframeKeeper(core::Call& c) {

    CallCinematicCamera *ccc = dynamic_cast<CallCinematicCamera*>(&c);
    if (ccc == NULL) return false;


    // CHECK for CHANGES in CinematicCamera CALL ------------------------------

    // Check for request of new interpolated keyframe
    if (ccc->changedInterpolatedKeyframeTime()) {
        ccc->setChangedInterpolatedKeyframeTime(false);

        this->interpolatedKeyframe = this->interpolateKeyframe(ccc->getInterpolatedKeyframeTime());
    }

    // Check for request of new selected keyframe
    if (ccc->changedSelectedKeyframeTime()) {
        ccc->setChangedSelectedKeyframeTime(false);

        float  time  = ccc->getSelectedKeyframeTime();
        int    index = -1; // vislib::Array<Keyframe>::INVALID_POS = -1

        // Check if requested selected keyframe is in keyframe array
        for (unsigned int i = 0; i < this->keyframes.Count(); i++) {
            if (time == this->keyframes[i].getTime()) {
                this->selectedKeyframe = this->keyframes[i];
                index = i;
                break;
            }
        }
        if (index == -1) {
            this->selectedKeyframe = this->interpolateKeyframe(time);
        }
    }

    // Check for new total time
    if (ccc->changedTotalTime()) {
        ccc->setChangedTotalTime(false);

        this->totalTime = ccc->getTotalTime();
        this->setTotalTimeParam.Param<param::FloatParam>()->SetValue(this->totalTime);
    }

    // Check for new camera parameters for selected keyframe
    if (ccc->changedCameraParameter()) {
        ccc->setChangedCameraParameter(false);
        this->cameraParam = ccc->getCameraParameter();
        // update parameter
        /* wrong data type
        this->editCurrentPosParam.Param<param::Vector3fParam>()->SetValue(this->cameraParam->Position());
        this->editCurrentLookAtParam.Param<param::Vector3fParam>()->SetValue(this->cameraParam->LookAt());
        this->editCurrentUpParam.Param<param::Vector3fParam>()->SetValue(this->cameraParam->Up());
        */
    }


    // UPDATE PARAMETERS-------------------------------------------------------

    // DEBUG
    // vislib::sys::Log::DefaultLog.WriteWarn("<what happened>.<what to do>");

    // addKeyframeParam
    if (this->addKeyframeParam.IsDirty()) {

        // update bounding box
        // add new keyframe on right position in array
        // boundingBox.GrowToPoint(keyframes[keyframes.Count() - 1].getCamPosition());
        // inCall->getCameraForNewKeyframe() !!!!
        // increase time for new keyframe of 1/10 of total time ?

        vislib::graphics::Camera c;
        if (!this->cameraParam.IsNull()) {
            c.Parameters()->SetPosition(this->cameraParam->Position());
            c.Parameters()->SetLookAt(this->cameraParam->LookAt());
            c.Parameters()->SetUp(this->cameraParam->Up());
            c.Parameters()->SetApertureAngle(this->cameraParam->ApertureAngle());
        }
        float t = this->timeForNewKeyframeParam.Param<param::FloatParam>()->Value();

        this->keyframes.Add(Keyframe(c, t));
        this->selectedKeyframe = this->keyframes.Last();
        this->boundingBox.GrowToPoint(this->keyframes.Last().getCamPosition());

      
        this->addKeyframeParam.ResetDirty();
    }

    // timeForNewKeyframeParam
    if (this->timeForNewKeyframeParam.IsDirty()) {

        // update selected keyframe ...


        this->timeForNewKeyframeParam.ResetDirty();
    }

    // deleteSelectedKeyframeParam
    if (this->deleteSelectedKeyframeParam.IsDirty()) {

        // check if it was the last keyframe
        // ccc: set seleected keyframe pointer to NULL


        this->deleteSelectedKeyframeParam.ResetDirty();
    }

    // editCurrentTimeParam
    if (this->editCurrentTimeParam.IsDirty()) {
        // CHANGE VALUE ONLY IF: !this->keyframes.IsEmpty()



        this->editCurrentTimeParam.ResetDirty();
    }

    // editCurrentPosParam
    if (this->editCurrentPosParam.IsDirty()) {
        // CHANGE VALUE ONLY IF: !this->keyframes.IsEmpty()



        this->editCurrentPosParam.ResetDirty();
    }

    // editCurrentLookAtParam
    if (this->editCurrentLookAtParam.IsDirty()) {
        // CHANGE VALUE ONLY IF: !this->keyframes.IsEmpty()



        this->editCurrentLookAtParam.ResetDirty();
    }

    // editCurrentUpParam
    if (this->editCurrentUpParam.IsDirty()) {
        // CHANGE VALUE ONLY IF: !this->keyframes.IsEmpty()



        this->editCurrentUpParam.ResetDirty();
    }

    // setTotalTimeParam
    if (this->setTotalTimeParam.IsDirty()) {
        float tt = this->setTotalTimeParam.Param<param::FloatParam>()->Value();
        if (!this->keyframes.IsEmpty()) {
            if (tt < this->keyframes.Last().getTime()) {
                tt = this->keyframes.Last().getTime();
                this->setTotalTimeParam.Param<param::FloatParam>()->SetValue(tt);
                vislib::sys::Log::DefaultLog.WriteWarn("KEYFRAME KEEPER [setTotalTimeParam] TOTAL TIME is smaller than time of last keyframe. Delete Keyframe(s) to reduce TOTAL TIME to desired value.");
            }
        }
        this->totalTime = tt;
        this->setTotalTimeParam.ResetDirty();
    }

    // fileNameParam
    if (this->fileNameParam.IsDirty()) {




        this->fileNameParam.ResetDirty();
    }

    // saveKeyframesParam
    if (this->saveKeyframesParam.IsDirty()) {




        this->saveKeyframesParam.ResetDirty();
    }

    // loadKeyframesParam
    if (this->loadKeyframesParam.IsDirty()) {




        this->loadKeyframesParam.ResetDirty();
    }

    // autoLoadKeyframesAtStartParam
    if (this->autoLoadKeyframesAtStartParam.IsDirty()) {




        this->autoLoadKeyframesAtStartParam.ResetDirty();
    }



    // PROPAGATE CHANGES TO CALL ----------------------------------------------

    ccc->setKeyframes(&this->keyframes);
    ccc->setBoundingBox(&this->boundingBox);
    ccc->setSelectedKeyframe(&this->selectedKeyframe);
    ccc->setInterpolatedKeyframe(&this->interpolatedKeyframe);
    ccc->setTotalTime(this->totalTime);


    return true;
}


/*
* KeyframeKeeper::interpolateKeyframe
*/
Keyframe KeyframeKeeper::interpolateKeyframe(float t) {

    // checking if keyframe for given time is already in keyframe array is intentionally omitted

    Keyframe retKeyframe;

    if (this->keyframes.IsEmpty()) {
        vislib::sys::Log::DefaultLog.WriteWarn("KEYFRAME KEEPER [interpolateKeyframe] INTERPOLATING KEYFRAME is not possible on empty keyframe array. Add some keyframes first.");
        return Keyframe();
    }
    else if (this->keyframes.Count() == 1) {
        vislib::sys::Log::DefaultLog.WriteWarn("KEYFRAME KEEPER [interpolateKeyframe] INTERPOLATING KEYFRAME is not possible for just one existing keyframe. Add some keyframes first.");
        return this->keyframes.First();
    }
    else if (t <= this->keyframes.First().getTime()) {
        //vislib::sys::Log::DefaultLog.WriteWarn("KEYFRAME KEEPER [interpolateKeyframe] INTERPOLATING KEYFRAME is not possible for time less than minimum time of keyframes. Add keyframe with lower time first.");
        return this->keyframes.First();
    }
    else if (t >= this->keyframes.Last().getTime()) {
        //vislib::sys::Log::DefaultLog.WriteWarn("KEYFRAME KEEPER [interpolateKeyframe] INTERPOLATING KEYFRAME is not possible for time greater than maximum time of keyframes. Add keyframe with greater time first.");
        return this->keyframes.Last();
    }
    else { // if ((t > this->keyframes.First().getTime()) && (t < this->keyframes.Last().getTime())) {
        // ASSUMPTION: there are at least two keyframes and the given time lies in between them

// TODO 

        // new default keyframe
        Keyframe k = Keyframe(vislib::graphics::Camera(), t);

        // determine indices of in interpolation involved keyframes
        int i0 = 0;
        int i1 = 0;
        int i2 = 0;
        int i3 = 0;
        int kfCnt = (int)keyframes.Count()-1;

        for (int i = 0; i < kfCnt; i++) {
            if ((this->keyframes[i].getTime() < t) && (t < this->keyframes[i + 1].getTime())) {
                i1 = i;
                i2 = i + 1;
                break;
            }
        }
        i0 = (i1 > 0) ? (i1 - 1) : (0);
        i3 = (i2 < kfCnt) ? (i2 + 1) : (kfCnt);

        //interpolate position
        vislib::math::Vector<float, 3> p0(keyframes[i0].getCamPosition());
        vislib::math::Vector<float, 3> p1(keyframes[i1].getCamPosition());
        vislib::math::Vector<float, 3> p2(keyframes[i2].getCamPosition());
        vislib::math::Vector<float, 3> p3(keyframes[i3].getCamPosition());

        vislib::math::Vector<float, 3> pk = (((p1 * 2.0f) +
            (p2 - p0) * t +
            (p0 * 2 - p1 * 5 + p2 * 4 - p3) * t * t +
            (-p0 + p1 * 3 - p2 * 3 + p3) * t * t * t) * 0.5);
        k.setCameraPosition(Point<float, 3>(pk.GetX(), pk.GetY(), pk.GetZ()));

        //interpolate lookAt
        vislib::math::Vector<float, 3> l0(keyframes[i0].getCamLookAt());
        vislib::math::Vector<float, 3> l1(keyframes[i1].getCamLookAt());
        vislib::math::Vector<float, 3> l2(keyframes[i2].getCamLookAt());
        vislib::math::Vector<float, 3> l3(keyframes[i3].getCamLookAt());

        vislib::math::Vector<float, 3> lk = (((l1 * 2) +
            (l2 - l0) * t +
            (l0 * 2 - l1 * 5 + l2 * 4 - l3) * t * t +
            (-l0 + l1 * 3 - l2 * 3 + l3) * t * t * t) * 0.5);
        k.setCameraLookAt(Point<float, 3>(lk.GetX(), lk.GetY(), lk.GetZ()));

        //interpolate up
        vislib::math::Vector<float, 3> u0 = p0 + keyframes[i0].getCamUp();
        vislib::math::Vector<float, 3> u1 = p1 + keyframes[i1].getCamUp();
        vislib::math::Vector<float, 3> u2 = p2 + keyframes[i2].getCamUp();
        vislib::math::Vector<float, 3> u3 = p3 + keyframes[i3].getCamUp();

        vislib::math::Vector<float, 3> uk = (((u1 * 2) +
            (u2 - u0) * t +
            (u0 * 2 - u1 * 5 + u2 * 4 - u3) * t * t +
            (-u0 + u1 * 3 - u2 * 3 + u3) * t * t * t) * 0.5);
        k.setCameraUp(uk - pk);

        //interpolate aperture angle
        float a0 = keyframes[i0].getCamApertureAngle();
        float a1 = keyframes[i1].getCamApertureAngle();
        float a2 = keyframes[i2].getCamApertureAngle();
        float a3 = keyframes[i3].getCamApertureAngle();

        a0 = (((a1 * 2) +
            (a2 - a0) * t +
            (a0 * 2 - a1 * 5 + a2 * 4 - a3) * t * t +
            (-a0 + a1 * 3 - a2 * 3 + a3) * t * t * t) * 0.5f);

        k.setCameraApertureAngele(a0);

        return k;
    }
}


/*
* KeyframeKeeper::loadKeyframes
*/
void KeyframeKeeper::loadKeyframes() {

    /*
    try {
    keyframes.Clear();
    std::ifstream myfile;
    myfile.open(fileNameParam.Param<param::FilePathParam>()->Value());

    vislib::StringSerialiserA ser;
    std::string line;
    vislib::StringA camera;
    int maxID = 0;
    // set totalTime
    std::getline(myfile, line);

    totalTimeParam.Param<param::FloatParam>()->SetValue(std::stof(line.erase(0, 10)));
    totalTimeParam.ResetDirty();

    int currentKeyframe = 0;
    while (std::getline(myfile, line)){

        if (line.substr(0, 5) == "time="){
            keyframes.Add(Keyframe());
            keyframes[currentKeyframe].setTime(static_cast<float>(std::stof(line.erase(0, 5))));
        }
        else if (line.empty()) {
            if (!camera.IsEmpty()) {
                // this line suddenly crashes (stack overflow). WHY?
                ser.SetInputString(camera);
                keyframes[currentKeyframe].getCamParameters()->Deserialise(ser);
                boundingBox.GrowToPoint(keyframes[currentKeyframe].getCamPosition());
                currentKeyframe++;
                camera.Clear();
            }

        }
        else {
            camera.Append(line.c_str());
            camera.Append("\n");
        }
    }

    selectedKeyframeIndex = 0;
    // ensure loaded Keyframes are sorted
    this->sortKeyframes();
    return true;
    }
    catch (...) {
    vislib::sys::Log::DefaultLog.WriteError("Loading keyframes failed!");
    return false;
    }
    */

}


/*
* KeyframeKeeper::saveKeyframes
*/
void KeyframeKeeper::saveKeyframes() {

    /*
    std::ofstream myfile;
    myfile.open(fileNameParam.Param<param::FilePathParam>()->Value(), std::ios::binary);
    vislib::StringSerialiserA ser;
    myfile << "totalTime=" << totalTimeParam.Param<param::FloatParam>()->Value() << "\n";
    for (unsigned int i = 0; i < keyframes.Count(); i++){
        keyframes[i].getCamParameters()->Serialise(ser);
        myfile << "time=" << keyframes[i].getTime() << "\n";
        myfile << ser.GetString().PeekBuffer() << "\n";
    }
    myfile.close();
    */

}
