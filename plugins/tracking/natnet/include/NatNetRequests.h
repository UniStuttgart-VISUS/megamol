//======================================================================================================
// Copyright 2016, NaturalPoint Inc.
//======================================================================================================

#pragma once


// Parameters: (none)
// Returns: float32_t value representing current system's unit scale in terms of millimeters.
#define NATNET_REQUEST_GETUNITSTOMILLIMETERS "UnitsToMillimeters"

// Parameters: (none)
// Returns: float32_t value representing current system's tracking frame rate in frames per second.
#define NATNET_REQUEST_GETFRAMERATE "FrameRate"

// Parameters: (none)
// Returns: int32_t value representing number of analog samples per mocap frame of data.
#define NATNET_REQUEST_GETANALOGSAMPLESPERMOCAPFRAME "AnalogSamplesPerMocapFrame"

// Parameters: (none)
// Returns: int32_t value representing length of current take in frames.
#define NATNET_REQUEST_GETCURRENTTAKELENGTH "CurrentTakeLength"

// Parameters: (none)
// Returns: int32_t value representing the current mode. (0 = Live, 1 = Recording, 2 = Edit)
#define NATNET_REQUEST_GETCURRENTMODE "CurrentMode"

// Parameters: (none)
// Returns: (none)
#define NATNET_REQUEST_STARTRECORDING "StartRecording"

// Parameters: (none)
// Returns: (none)
#define NATNET_REQUEST_STOPRECORDING "StopRecording"

// Parameters: (none)
// Returns: (none)
#define NATNET_REQUEST_SWITCHTOLIVEMODE "LiveMode"

// Parameters: (none)
// Returns: (none)
#define NATNET_REQUEST_SWITCHTOEDITMODE "EditMode"

// Parameters: (none)
// Returns: (none)
#define NATNET_REQUEST_TIMELINEPLAY "TimelinePlay"

// Parameters: (none)
// Returns: (none)
#define NATNET_REQUEST_TIMELINESTOP "TimelineStop"

// Parameters: New playback take name.
// Returns: (none)
#define NATNET_REQUEST_SETPLAYBACKTAKENAME "SetPlaybackTakeName"

// Parameters: New record take name.
// Returns: (none)
#define NATNET_REQUEST_SETRECORDTAKENAME "SetRecordTakeName"

// Parameters: Session name to either switch to or to create.
// Returns: (none)
#define NATNET_REQUEST_SETCURRENTSESSION "SetCurrentSession"

// Parameters: The new start frame for the playback range.
// Returns: (none)
#define NATNET_REQUEST_SETPLAYBACKSTARTFRAME "SetPlaybackStartFrame"

// Parameters: The new end frame for the playback range.
// Returns: (none)
#define NATNET_REQUEST_SETPLAYBACKSTOPFRAME "SetPlaybackStopFrame"

// Parameters: The new current frame for playback.
// Returns: (none)
#define NATNET_REQUEST_SETPLAYBACKCURRENTFRAME "SetPlaybackCurrentFrame"

// Parameters: Node (asset) name
// Returns: int32_t value indicating whether the command successfully enabled the asset. 0 if succeeded.
// (e.g. string command = "EnableAsset,Rigid Body 1")
#define NATNET_REQUEST_ENABLEASSET "EnableAsset"

// Parameters: Node (asset) name
// Returns: int32_t value indicating whether the command successfully disabled the asset. 0 if succeeded.
#define NATNET_REQUEST_DISABLEASSET "DisableAsset"

// Parameters: Node name (if applicable leave it empty if not) and property name. 
// Returns: string value representing corresponding property settings
#define NATNET_REQUEST_GETPROPERTY "GetProperty"

// Parameters: Node name (if applicable leave it empty if not), property name, and desired value.
// Returns: int32_t value indicating whether the command successfully updated the data. 0 if succeeded.
// (e.g. string command = "SetProperty,,Unlabeled Markers,False")
#define NATNET_REQUEST_SETPROPETRY "SetProperty"

