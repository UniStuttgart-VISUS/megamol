/*
* KeyframeKeeper.cpp
*
*/
#include "stdafx.h"
#include "KeyframeKeeper.h"
#include "CallCinematicCamera.h"
#include "vislib/math/Point.h"
#include "vislib/math/Vector.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/AbstractSlot.h"
#include "mmcore/utility/xml/XmlParser.h"
#include "mmcore/utility/xml/XmlReader.h"
#include "vislib/StringSerialiser.h"
#include <iostream>
#include <fstream>

using namespace megamol;
using namespace megamol::cinematiccamera;
using namespace vislib::math;
using namespace megamol::core;

//TODO: Load/Save/Filename params


/*
* KeyframeKeeper::KeyframeKeeper
*/
KeyframeKeeper::KeyframeKeeper(void) : core::Module(),
	cinematicRendererSlot("scene3D", "Sends data to the 3D Scene"), 
	saveKeyframesParam("save", "Serializes and saves the Keyframes"),
	loadKeyframesParam("load", "Deserializes and loads the Keyframes"),
	fileNameParam("filename", "The name of the file to load or save"),
	totalTime("totalTime", "The total timespan of the movie in seconds"),
	selectedKeyframe("selectedKeyframe", "the index of the selected Keyframe"),
	addKeyframeAtSelectedPosition("addKeyframeAtSelectedPosition", "adds a Keyframe at the currently selected position inbetween two keyframes"),
	currentKeyframeTime("currentKeyframeTime", "timestamp of the currently selected keyframe"),
	currentPos("currentPos", "Position of the selected keyframe"),
	currentLookAt("currentLookAt", "LookAt of the selected keyframe"),
	currentUp("currentUp", "LookAt of the selected keyframe"),
	keyframes(), selectedKeyframeIndex(0) {

	// setting up callbacks for the Cinematic renderer
	this->cinematicRendererSlot.SetCallback(CallCinematicCamera::ClassName(),
		CallCinematicCamera::FunctionName(CallCinematicCamera::CallForGetKeyframes), &KeyframeKeeper::cbAllKeyframes);

	this->cinematicRendererSlot.SetCallback(CallCinematicCamera::ClassName(),
		CallCinematicCamera::FunctionName(CallCinematicCamera::CallForSelectKeyframe), &KeyframeKeeper::cbSelectKeyframe);

	this->cinematicRendererSlot.SetCallback(CallCinematicCamera::ClassName(),
		CallCinematicCamera::FunctionName(CallCinematicCamera::CallForGetSelectedKeyframe), &KeyframeKeeper::cbSelectedKeyframe);

	this->cinematicRendererSlot.SetCallback(CallCinematicCamera::ClassName(),
		CallCinematicCamera::FunctionName(CallCinematicCamera::CallForInterpolatedKeyframe), &KeyframeKeeper::cbInterpolateKeyframe);

	this->cinematicRendererSlot.SetCallback(CallCinematicCamera::ClassName(),
		CallCinematicCamera::FunctionName(CallCinematicCamera::CallForGetTotalTime), &KeyframeKeeper::cbGetTotalTime);

	this->cinematicRendererSlot.SetCallback(CallCinematicCamera::ClassName(),
		CallCinematicCamera::FunctionName(CallCinematicCamera::CallForGetKeyframeAtTime), &KeyframeKeeper::cbGetKeyframeAtTime);

	this->cinematicRendererSlot.SetCallback(CallCinematicCamera::ClassName(),
		CallCinematicCamera::FunctionName(CallCinematicCamera::CallForNewKeyframeAtPosition), &KeyframeKeeper::cbNewKeyframe);

	this->MakeSlotAvailable(&this->cinematicRendererSlot);

	this->saveKeyframesParam.SetParameter(new param::ButtonParam());
	this->saveKeyframesParam.SetUpdateCallback(this, &KeyframeKeeper::cbSave);
	this->MakeSlotAvailable(&this->saveKeyframesParam);

	
	this->loadKeyframesParam.SetParameter(new param::ButtonParam());
	this->loadKeyframesParam.SetUpdateCallback(this, &KeyframeKeeper::cbLoad);
	this->MakeSlotAvailable(&this->loadKeyframesParam);

	this->fileNameParam.SetParameter(new param::FilePathParam("enter filename here"));
	this->MakeSlotAvailable(&this->fileNameParam);

	this->totalTime.SetParameter(new param::FloatParam(1));
	this->MakeSlotAvailable(&this->totalTime);
	prevTotalTime = 1;

	this->selectedKeyframe.SetParameter(new param::FloatParam(-1));
	this->MakeSlotAvailable(&this->selectedKeyframe);

	this->addKeyframeAtSelectedPosition.SetParameter(new param::ButtonParam());
	this->addKeyframeAtSelectedPosition.SetUpdateCallback(this, &KeyframeKeeper::cbAddKeyframeAtSelectedPosition);
	this->MakeSlotAvailable(&this->addKeyframeAtSelectedPosition);

	keyframes.Add(Keyframe(vislib::graphics::Camera(), 0.0f, -2));

	this->currentKeyframeTime.SetParameter(new param::FloatParam(keyframes[(int)selectedKeyframeIndex].getTime()*totalTime.Param<param::FloatParam>()->Value()));
	this->MakeSlotAvailable(&this->currentKeyframeTime);

	this->currentPos.SetParameter(new param::Vector3fParam(vislib::math::Vector<float, 3>(keyframes[(int)selectedKeyframeIndex].getCamPosition())));
	this->MakeSlotAvailable(&this->currentPos);

	this->currentLookAt.SetParameter(new param::Vector3fParam(vislib::math::Vector<float, 3>(keyframes[(int)selectedKeyframeIndex].getCamLookAt())));
	this->MakeSlotAvailable(&this->currentLookAt);

	this->currentUp.SetParameter(new param::Vector3fParam(vislib::math::Vector<float, 3>(keyframes[(int)selectedKeyframeIndex].getCamUp())));
	this->MakeSlotAvailable(&this->currentUp);

	IDCounter = 0;
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
	// intentionally empty
	return true;
}

/*
* KeyframeKeeper::release(void)
*/
void KeyframeKeeper::release(void) {
	// intentionally empty
}

/** Callback for getting all Keyframes*/
bool KeyframeKeeper::cbAllKeyframes(core::Call& c){
	updateParameters();
	CallCinematicCamera *inCall = dynamic_cast<CallCinematicCamera*>(&c);
	if (inCall == NULL) return false;

	inCall->setBoundingBox(&this->boundingBox);
	inCall->setKeyframes(&this->keyframes);
	inCall->setTotalTime(totalTime.Param<param::FloatParam>()->Value());
	return true;
}

/** Callback for getting selected Keyframe*/
bool KeyframeKeeper::cbSelectedKeyframe(core::Call& c){
	updateParameters();
	CallCinematicCamera *inCall = dynamic_cast<CallCinematicCamera*>(&c);
	if (inCall == NULL) return false;
	if (this->keyframes.IsEmpty()) return false;
	inCall->setKeyframes(&this->keyframes);

	if (this->selectedKeyframeIndex < 0 || this->selectedKeyframeIndex >= this->keyframes.Count()) return false;
	inCall->setSelectedKeyframeIndex(this->selectedKeyframeIndex);
	
	if (ceilf(selectedKeyframeIndex) != selectedKeyframeIndex){
		inCall->setInterpolatedKeyframe(interpolateKeyframe(selectedKeyframeIndex));
	}

	inCall->setBoundingBox(&this->boundingBox);

	return true;
}


/** Callback for selecting a new Keyframe */
bool KeyframeKeeper::cbSelectKeyframe(core::Call& c){
	CallCinematicCamera *inCall = dynamic_cast<CallCinematicCamera*>(&c);
	if (inCall == NULL) return false;
	
	this->selectedKeyframeIndex = inCall->getSelectedKeyframeIndex();

	selectedKeyframe.Param<param::FloatParam>()->SetValue(selectedKeyframeIndex);
	
	if (ceilf(selectedKeyframeIndex) != selectedKeyframeIndex){
		inCall->setInterpolatedKeyframe(interpolateKeyframe(selectedKeyframeIndex));
		currentKeyframeTime.Param<param::FloatParam>()->SetValue(inCall->getInterpolatedKeyframe().getTime()*totalTime.Param<param::FloatParam>()->Value());
		currentPos.Param<param::Vector3fParam>()->SetValue(inCall->getInterpolatedKeyframe().getCamPosition());
		currentLookAt.Param<param::Vector3fParam>()->SetValue(inCall->getInterpolatedKeyframe().getCamLookAt());
		currentUp.Param<param::Vector3fParam>()->SetValue(inCall->getInterpolatedKeyframe().getCamUp());
	}
	else {
		currentKeyframeTime.Param<param::FloatParam>()->SetValue(keyframes[(int)(selectedKeyframeIndex)].getTime()*totalTime.Param<param::FloatParam>()->Value());
		currentPos.Param<param::Vector3fParam>()->SetValue(keyframes[(int)ceil(selectedKeyframeIndex)].getCamPosition());
		currentUp.Param<param::Vector3fParam>()->SetValue(keyframes[(int)ceil(selectedKeyframeIndex)].getCamUp());
	}

	currentKeyframeTime.ResetDirty();
	currentPos.ResetDirty();
	currentLookAt.ResetDirty();
	currentUp.ResetDirty();
	selectedKeyframe.ResetDirty();

	return true;
}

/** Callback for deleting selected Keyframe */
bool KeyframeKeeper::cbDelKeyframe(core::Call& c){
	CallCinematicCamera *inCall = dynamic_cast<CallCinematicCamera*>(&c);
	if (inCall == NULL) return false;

	this->keyframes.RemoveAt(static_cast<SIZE_T>(inCall->getSelectedKeyframeIndex()));
	// decrease selectedKeyframeIndex by 1 or set to 0 if it would become negative
	this->selectedKeyframeIndex = (0 < inCall->getSelectedKeyframeIndex() - 1 ? inCall->getSelectedKeyframeIndex() - 1 : 0);
	return true;
}

/** Callback for interpolating Keyframe */
bool KeyframeKeeper::cbInterpolateKeyframe(core::Call& c){
	CallCinematicCamera *inCall = dynamic_cast<CallCinematicCamera*>(&c);
	if (inCall == NULL){
		vislib::sys::Log::DefaultLog.WriteError("inCall was NULL at cbInterpolateKeyframe!");
		return false;
	}
	inCall->setInterpolatedKeyframe(interpolateKeyframe(inCall->getIndexToInterPolate()));
	float idx = inCall->getIndexToInterPolate();
	
	return true;
	
}


/** Callback for creating new Keyframe */
bool KeyframeKeeper::cbNewKeyframe(core::Call& c){
	CallCinematicCamera *inCall = dynamic_cast<CallCinematicCamera*>(&c);
	if (inCall == NULL){
		vislib::sys::Log::DefaultLog.WriteError("inCall was NULL at cbNewKeyframe!");
		return false;
	}
	if (keyframes.Count() == 0){
		keyframes.Add(Keyframe(inCall->getCameraForNewKeyframe(), 0.0f, IDCounter));
		IDCounter++;
	}
	else{
		keyframes.Add(Keyframe(inCall->getCameraForNewKeyframe(), keyframes[keyframes.Count() - 1].getTime() + 0.1f, IDCounter));
		IDCounter++;
	}
	ASSERT(keyframes.Count() >= 1);
	boundingBox.GrowToPoint(keyframes[keyframes.Count() - 1].getCamPosition());
	sortKeyframes();
	return true;
}

/** Callback for getting total time */
bool KeyframeKeeper::cbGetTotalTime(core::Call& c){
	CallCinematicCamera *inCall = dynamic_cast<CallCinematicCamera*>(&c);
	if (inCall == NULL){
		vislib::sys::Log::DefaultLog.WriteError("inCall was NULL at cbGetTotalTime!");
		return false;
	}
	inCall->setTotalTime(totalTime.Param<param::FloatParam>()->Value());
	return true;
}

/** Callback for getting the keyframe at a time bookmarked in the call*/
bool KeyframeKeeper::cbGetKeyframeAtTime(core::Call& c){
	//ignore dummy keyframe
	if (keyframes.Count() <= selectedKeyframeIndex) return false;
	if (keyframes[(int)selectedKeyframeIndex].getID() == -2) return false;

	CallCinematicCamera *inCall = dynamic_cast<CallCinematicCamera*>(&c);
	
	if (inCall == NULL){
		vislib::sys::Log::DefaultLog.WriteError("inCall was NULL at cbGetKeyframeAtTime!");
		return false;
	}
	float time = inCall->getTimeofKeyframeToGet();

	int i = 0;
	// search keyframes until time is keyframe with a greater time is found
	while (keyframes[i].getTime()*totalTime.Param<param::FloatParam>()->Value() < time){
		if (static_cast<SIZE_T>(i + 1) < keyframes.Count()){
			i++;
		}
		else{
			vislib::sys::Log::DefaultLog.WriteWarn("Trajectory ended. Rendering last position");
			inCall->setInterpolatedKeyframe(keyframes[i]);
			return true;
		}
	}
	// time matches the time of an exact keyframe
	ASSERT(keyframes.Count() > static_cast<SIZE_T>(i));
	if (keyframes[i].getTime()*totalTime.Param<param::FloatParam>()->Value() == time){
		inCall->setInterpolatedKeyframe(keyframes[i]);
		return true;
	}
	else{
		// index of wanted keyframe is
		// i - (i'th Keyframe time - time) / (i'th Keyframe time - i-1'th Keyframe time)
		// with denormalized Keyframe times
		inCall->setInterpolatedKeyframe(interpolateKeyframe(i - 
			((keyframes[i].getTime()*totalTime.Param<param::FloatParam>()->Value() - time) /
			(keyframes[i].getTime()*totalTime.Param<param::FloatParam>()->Value() - keyframes[i - 1].getTime()*totalTime.Param<param::FloatParam>()->Value()))));
		return true;
	}
}

/** Callback for the save button */
bool KeyframeKeeper::cbSave(core::param::ParamSlot& slot){
	//ensure saved Keyframes are sorted
	sortKeyframes();

	std::ofstream myfile;
	myfile.open(fileNameParam.Param<param::FilePathParam>()->Value());
	vislib::StringSerialiserA ser;
	myfile << "totalTime=" << totalTime.Param<param::FloatParam>()->Value() << std::endl;
	for (unsigned int i = 0; i < keyframes.Count(); i++){
		keyframes[i].getCamParameters()->Serialise(ser);
		myfile << "time=" << keyframes[i].getTime() << std::endl;
		myfile << "ID=" << keyframes[i].getID() << std::endl;
		myfile << ser.GetString().PeekBuffer() << std::endl;
	}
	myfile.close();

	return true;
}

/** Callback for the load button */
bool KeyframeKeeper::cbLoad(core::param::ParamSlot& slot){
	int i = 1;
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
		totalTime.Param<param::FloatParam>()->SetValue(std::stof(line.erase(0, 10)));
		totalTime.ResetDirty();

		int currentKeyframe = 0;
		while (std::getline(myfile, line)){
			if (line.substr(0, 5) == "time="){
				keyframes.Add(Keyframe());
				keyframes[currentKeyframe].setTime(static_cast<float>(std::stof(line.erase(0, 5))));
			}
			else if (line.substr(0, 3) == "ID="){
				keyframes[currentKeyframe].setID(static_cast<int>(std::stof(line.erase(0, 3))));
				maxID = maxID > keyframes[currentKeyframe].getID() ? maxID : keyframes[currentKeyframe].getID();
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
		IDCounter = maxID + 1;
		selectedKeyframeIndex = 0;
		// ensure loaded Keyframes are sorted
		sortKeyframes();
		return true;
	}
	catch (...) {
		vislib::sys::Log::DefaultLog.WriteError("It did not work!");
		return false;
	}
}

/** Callback for inserting a keyframe at the currently selected position */
bool KeyframeKeeper::cbAddKeyframeAtSelectedPosition(core::param::ParamSlot& slot){
	if (ceilf(selectedKeyframeIndex) == selectedKeyframeIndex) {
		vislib::sys::Log::DefaultLog.WriteWarn("Did not insert Keyframe since no interpolated Keyframe was selected.");
		return false;
	}
	else{
		Keyframe k = interpolateKeyframe(selectedKeyframeIndex);
		k.setID(IDCounter);
		IDCounter++;
		keyframes.Add(k);
		selectedKeyframe.Param<param::FloatParam>()->SetValue(keyframes.Count() - 1.0f);
		ASSERT(keyframes.Count() >= 1);
		boundingBox.GrowToPoint(keyframes[keyframes.Count() - 1].getCamPosition());
		sortKeyframes();
	}
	return true;
}

void KeyframeKeeper::updateParameters(){
	if (selectedKeyframe.IsDirty()){
		if (selectedKeyframe.Param<param::FloatParam>()->Value() <= (float)keyframes.Count() - 1 && selectedKeyframe.Param<param::FloatParam>()->Value() >= 0){
			selectedKeyframeIndex = selectedKeyframe.Param<param::FloatParam>()->Value();
			currentKeyframeTime.Param<param::FloatParam>()->SetValue(interpolateKeyframe(selectedKeyframeIndex).getTime()*totalTime.Param<param::FloatParam>()->Value());
			currentKeyframeTime.ResetDirty();
		}
		else {
			vislib::sys::Log::DefaultLog.WriteWarn("Do not enter a Keyframeindex out of bounds you madman!");
			selectedKeyframe.Param<param::FloatParam>()->SetValue(selectedKeyframeIndex);
		}
		selectedKeyframe.ResetDirty();
	}

	if (ceilf(selectedKeyframeIndex) == selectedKeyframeIndex){
		if (currentPos.IsDirty()){
			keyframes[(int)selectedKeyframeIndex].setCameraPosition(
				vislib::math::Point<float, 3>(currentPos.Param<param::Vector3fParam>()->Value().GetX(),
											currentPos.Param<param::Vector3fParam>()->Value().GetY(),
											currentPos.Param<param::Vector3fParam>()->Value().GetZ()));
			boundingBox.GrowToPoint(keyframes[(int)selectedKeyframeIndex].getCamPosition());
			currentPos.ResetDirty();
		}
		if (currentLookAt.IsDirty()){
			keyframes[(int)selectedKeyframeIndex].setCameraLookAt(
				vislib::math::Point<float, 3>(currentLookAt.Param<param::Vector3fParam>()->Value().GetX(),
												currentLookAt.Param<param::Vector3fParam>()->Value().GetY(),
												currentLookAt.Param<param::Vector3fParam>()->Value().GetZ()));
			currentLookAt.ResetDirty();
		}
		if (currentKeyframeTime.IsDirty()){
			keyframes[(int)selectedKeyframeIndex].setTime(currentKeyframeTime.Param<param::FloatParam>()->Value()
															/ totalTime.Param<param::FloatParam>()->Value());
			currentKeyframeTime.ResetDirty();
			sortKeyframes();
		}
		if (currentUp.IsDirty()) {
			keyframes[(int)selectedKeyframeIndex].setCameraLookAt(
				vislib::math::Point<float, 3>(currentUp.Param<param::Vector3fParam>()->Value().GetX(),
					currentUp.Param<param::Vector3fParam>()->Value().GetY(),
					currentUp.Param<param::Vector3fParam>()->Value().GetZ()));
			currentUp.ResetDirty();
		}
	}
	if (totalTime.IsDirty()){
		// scaling the normalized time of the keyframes to fit the new total Time
		for (unsigned int i = 0; i < keyframes.Count(); i++){
			keyframes[i].setTime(keyframes[i].getTime()*prevTotalTime / totalTime.Param<param::FloatParam>()->Value());
		}
		totalTime.ResetDirty();
		prevTotalTime = totalTime.Param<param::FloatParam>()->Value();
	}
}

Keyframe KeyframeKeeper::interpolateKeyframe(float idx){

	if (idx > 0 && idx < keyframes.Count() - 1) {
		if (ceilf(idx) == idx) {
			return keyframes[(int)idx];
		}
		else {

			Keyframe k = Keyframe::Keyframe(-1);

			float t = idx - floorf(idx);

			//interpolate time
			k.setTime(keyframes[(int)floorf(idx)].getTime() + (keyframes[(int)ceilf(idx)].getTime() - keyframes[(int)floorf(idx)].getTime()) * t);


			//interpolate position
			vislib::math::Vector<float, 3> p0(keyframes[(0 > ((int)floor(idx) - 1)) ? (0) : ((int)floor(idx) - 1)].getCamPosition());
			vislib::math::Vector<float, 3> p1(keyframes[(int)floor(idx)].getCamPosition());
			vislib::math::Vector<float, 3> p2(keyframes[(int)ceil(idx)].getCamPosition());
			vislib::math::Vector<float, 3> p3(keyframes[(static_cast<int>(keyframes.Count()) - 1 < ((int)ceil(idx) + 1)) ? (static_cast<int>(keyframes.Count()) - 1) : ((int)ceil(idx) + 1)].getCamPosition());

			p0 = (((p1 * 2) +
				(p2 - p0) * t +
				(p0 * 2 - p1 * 5 + p2 * 4 - p3) * t * t +
				(-p0 + p1 * 3 - p2 * 3 + p3) * t * t * t) * 0.5);
			Point<float, 3> p(p0.GetX(), p0.GetY(), p0.GetZ());
			k.setCameraPosition(p);

			//interpolate lookAt
			p0 = keyframes[(0 > (int)floor(idx) - 1) ? (0) : ((int)floor(idx))].getCamLookAt();
			p1 = keyframes[(int)floor(idx)].getCamLookAt();
			p2 = keyframes[(int)ceil(idx)].getCamLookAt();
			p3 = keyframes[(static_cast<int>(keyframes.Count()) < (int)ceil(idx) + 1) ? (static_cast<int>(keyframes.Count())) : ((int)ceil(idx))].getCamLookAt();

			p0 = (((p1 * 2) +
				(p2 - p0) * t +
				(p0 * 2 - p1 * 5 + p2 * 4 - p3) * t * t +
				(-p0 + p1 * 3 - p2 * 3 + p3) * t * t * t) * 0.5);
			p.Set(p0.GetX(), p0.GetY(), p0.GetZ());
			k.setCameraLookAt(p);

			//interpolate up
			p0 = keyframes[(0 > (int)floor(idx) - 1) ? (0) : ((int)floor(idx))].getCamUp();
			p1 = keyframes[(int)floor(idx)].getCamUp();
			p2 = keyframes[(int)ceil(idx)].getCamUp();
			p3 = keyframes[(static_cast<int>(keyframes.Count()) < (int)ceil(idx) + 1) ? (static_cast<int>(keyframes.Count())) : ((int)ceil(idx))].getCamUp();

			p0 = (((p1 * 2) +
				(p2 - p0) * t +
				(p0 * 2 - p1 * 5 + p2 * 4 - p3) * t * t +
				(-p0 + p1 * 3 - p2 * 3 + p3) * t * t * t) * 0.5);
			p.Set(p0.GetX(), p0.GetY(), p0.GetZ());
			k.setCameraUp(p);

			//interpolate aperture angle
			float a0 = keyframes[(0 > (int)floor(idx) - 1) ? (0) : ((int)floor(idx))].getCamApertureAngle();
			float a1 = keyframes[(int)floor(idx)].getCamApertureAngle();
			float a2 = keyframes[(int)ceil(idx)].getCamApertureAngle();
			float a3 = keyframes[(static_cast<int>(keyframes.Count()) < (int)ceil(idx) + 1) ? (static_cast<int>(keyframes.Count())) : ((int)ceil(idx))].getCamApertureAngle();

			a0 = (((a1 * 2) +
				(a2 - a0) * t +
				(a0 * 2 - a1 * 5 + a2 * 4 - a3) * t * t +
				(-a0 + a1 * 3 - a2 * 3 + a3) * t * t * t) * 0.5f);
			
			k.setCameraApertureAngele(a0);

			return k;
		}
	}
	else {
		//requested a keyframe out of bounds
		return keyframes.First();
	}
}

void KeyframeKeeper::sortKeyframes(){
	int selectedKeyframeID;

	if (ceilf(selectedKeyframeIndex) == selectedKeyframeIndex) {
		if (selectedKeyframeIndex != -1){
			selectedKeyframeID = keyframes[(int)selectedKeyframeIndex].getID();
		}
	}

	//sorting by time
	//bubbleSort should be fast because array is usually almost sorted
	bool swapped;
	do {
		swapped = false;
		for (unsigned int i = 0; i < keyframes.Count() - 1; i++){
			if (keyframes[i].getTime() > keyframes[i + 1].getTime()){
				Keyframe temp = keyframes[i];
				keyframes[i] = keyframes[i + 1];
				keyframes[i + 1] = temp;
				swapped = true;
			}
		}
	} while (swapped);

	if (ceilf(selectedKeyframeIndex) == selectedKeyframeIndex) {
		if (selectedKeyframeIndex != -1){
			for (unsigned int i = 0; i < keyframes.Count(); i++){
				if (keyframes[i].getID() == selectedKeyframeID){
					selectedKeyframeIndex = static_cast<float>(i);
				}
			}
		}
	}
}