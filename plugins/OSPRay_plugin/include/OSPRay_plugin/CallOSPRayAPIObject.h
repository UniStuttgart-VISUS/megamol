/*
* CallOSPRayAPIObject.h
*
* Copyright (C) 2017 by Universitaet Stuttgart (VISUS).
* Alle Rechte vorbehalten.
*/

#pragma once

#include "mmcore/factories/CallAutoDescription.h"
#include "OSPRay_plugin/OSPRay_plugin.h"
#include "OSPRay_plugin/CallOSPRayStructure.h"

namespace megamol {
namespace ospray {

class OSPRAY_PLUGIN_API CallOSPRayAPIObject : public core::Call{
public:

    /**
    * Nested class interface for data unlockers. If data is returned with
    * an unlocker set. The caller must call 'Unlock' on the unlocker as
    * soon as the data is no longer required.
    */
    class Unlocker {
    public:

        /** ctor. */
        Unlocker(void) {
            // intentionally empty
        }

        /** dtor. */
        virtual ~Unlocker(void) {
            // intentionally empty
        }

        /** Unlocks the data */
        virtual void Unlock(void) = 0;

    };


    /**
    * Answer the name of the objects of this description.
    *
    * @return The name of the objects of this description.
    */
    static const char *ClassName(void) {
        return "CallOSPRayAPIObject";
    }

    /**
    * Gets a human readable description of the module.
    *
    * @return A human readable description of the module.
    */
    static const char *Description(void) {
        return "Call for an OSPRay API object";
    }

    /**
    * Answer the number of functions used for this call.
    *
    * @return The number of functions used for this call.
    */
    static unsigned int FunctionCount(void) {
        return 3;
    }

    /**
    * Answer the name of the function used for this call.
    *
    * @param idx The index of the function to return it's name.
    *
    * @return The name of the requested function.
    */
    static const char * FunctionName(unsigned int idx) {
        switch (idx) {
        case 0: return "GetData";
        case 1: return "GetExtent";
        case 2: return "GetDirty";
        default: return NULL;
        }
    }

    /**
    * Answers the bounding box of the data set
    *
    * @return The bounding box of the data set
    */
    inline core::BoundingBoxes_2& AccessBoundingBoxes(void) {
        return this->bboxs;
    }

    /**
    * Answer the frame count.
    *
    * @return The frame count.
    */
    inline unsigned int FrameCount(void) const {
        return this->frameCnt;
    }

    /**
    * Answer the frameID
    *
    * @return the frameID
    */
    inline unsigned int FrameID(void) const {
        return this->frameID;
    }

    /**
    * Answers the bounding box of the data set
    *
    * @return The bounding box of the data set
    */
    inline const core::BoundingBoxes_2& GetBoundingBoxes(void) const {
        return this->bboxs;
    }

    /**
    * Answer if the frame is forced. If 'true' a data provider must
    * return exactly the requested frame, no matter how long the loading
    * takes. If 'false' a data provider should return as fast as possible
    * returning the closest match to the requested data (updating the
    * frameID attribute).
    *
    * @return The flag if the frame is forced
    */
    inline bool IsFrameForced(void) const {
        return this->forceFrame;
    }

    /**
    * Sets the extents of the data.
    * Called modules uses this method to output their data.
    *
    * @param frameCnt The number of frames in the trajectory.
    * @param minX The minimum x coordinate
    * @param minY The minimum y coordinate
    * @param minZ The minimum z coordinate
    * @param maxX The maximum x coordinate
    * @param maxY The maximum y coordinate
    * @param maxZ The maximum z coordinate
    */
    inline void SetExtent(unsigned int frameCnt,
        float minX, float minY, float minZ,
        float maxX, float maxY, float maxZ) {
        this->frameCnt = frameCnt;
        this->bboxs.Clear();
        this->bboxs.SetBoundingBox(minX, minY, minZ, maxX, maxY, maxZ);
        this->bboxs.SetClipBox(minX, minY, minZ, maxX, maxY, maxZ);
    }

    /**
    * Sets the extents of the data.
    * Called modules uses this method to output their data.
    *
    * @param frameCnt The number of frames in the trajectory.
    * @param bboxs The bounding boxes of the data
    */
    inline void SetExtent(unsigned int frameCnt,
        const core::BoundingBoxes_2& bboxs) {
        this->frameCnt = frameCnt;
        this->bboxs = bboxs;
    }

    /**
    * Sets the number of time frames of the data
    *
    * @param frameCnt The number of frames in the trajectory.
    */
    inline void SetFrameCount(unsigned int frameCnt) {
        this->frameCnt = frameCnt;
    }

    /**
    * Sets the frameID to request data for.
    *
    * @param frameID The frameID to request data for.
    * @param force Flag whether or not to force the frame data. If the
    *              data is not forces, a source module may return data
    *              from any other frame (the best data currently
    *              available). If the data is forced, the call might not
    *              return until the data is loaded.
    */
    inline void SetFrameID(unsigned int frameID, bool force = false) {
        this->forceFrame = force;
        this->frameID = frameID;
    }

    /**
    * Assignment operator.
    * Makes a deep copy of all members. While for data these are only
    * pointers, the pointer to the unlocker object is also copied.
    *
    * @param rhs The right hand side operand
    *
    * @return A reference to this
    */
    CallOSPRayAPIObject& operator=(const CallOSPRayAPIObject& rhs);

    /**
    * Answer the unique hash number of the returned data, or zero if such
    * a number can not be provided.
    *
    * @return The unique hash number of the returned data
    */
    inline SIZE_T DataHash(void) const {
        return this->datahash;
    }

    /**
    * Answer the unlocker
    *
    * @return The unlocker
    */
    inline Unlocker *GetUnlocker(void) const {
        return this->unlocker;
    }

    /**
    * Sets the unique hash number for the returned data, or zero if such
    * a number can not be provided.
    *
    * @param hash The unique hash number
    */
    inline void SetDataHash(SIZE_T hash) {
        this->datahash = hash;
    }

    /**
    * Sets the data unlocker and optionally unlocks the old data if
    * present. The memory of the unlocker object 'unlocker' must be
    * allocated with defaut 'new'. The object will be owned by this call
    * object after this method returns. This means the caller must not
    * change the 'unlocker' object anymore, especially he must not delete
    * the object.
    *
    * @param unlocker The new unlocker object to use.
    * @param unlockOld If 'true' 'Unlock' is called before the new
    *                  unlocker is set.
    */
    inline void SetUnlocker(Unlocker *unlocker, bool unlockOld = true) {
        if (unlockOld) this->Unlock();
        this->unlocker = unlocker;
    }

    /**
    * Unlocks the data stored
    * This must be called after the data is no longer used to avoid
    * deadlocks in the out-of-core streaming mechanism.
    */
    inline void Unlock(void) {
        if (this->unlocker != NULL) {
            this->unlocker->Unlock();
            SAFE_DELETE(this->unlocker);
        }
    }

    /**
    * Gets the data defined time stamp
    *
    * @return The data defined time stamp
    */
    inline float GetTimeStamp(void) const {
        return timeStamp;
    }

    /**
    * Sets the data defined time stamp
    *
    * @param timeStamp The new time stamp value
    */
    void SetTimeStamp(float timeStamp) {
        this->timeStamp = timeStamp;
    }

    /** Ctor. */
    CallOSPRayAPIObject();

    /** Dtor. */
    virtual ~CallOSPRayAPIObject(void);

    void setAPIObjects(std::vector<void*> api_obj);
    std::vector<void*> getAPIObjects();

    void setStructureType(structureTypeEnum strtype);
    structureTypeEnum getStructureType();

    void resetDirty();
    void setDirty();
    bool isDirty();

private:
    std::vector<void*> api_obj;
    float timeStamp;
    structureTypeEnum type;
    bool dirtyFlag;

    /** Flag whether or not to force the frame data */
    bool forceFrame;

    /** The number of frames in the trajectory */
    unsigned int frameCnt;

    /** The requested/stored frameID */
    unsigned int frameID;

    /** the coordinate extents */
    core::BoundingBoxes_2 bboxs;

    /**
    * A unique hash number of the returned data, or zero if such a number
    * can not be provided
    */
    SIZE_T datahash;

    /** the data unlocker */
    Unlocker *unlocker;


};
typedef core::factories::CallAutoDescription<CallOSPRayAPIObject> CallOSPRayAPIObjectDescription;

} // namespace ospray
} // namespace megamol
