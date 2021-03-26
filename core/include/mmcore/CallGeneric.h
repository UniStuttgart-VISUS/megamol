/*
 * CallGeneric.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef CALL_GENERIC_H_INCLUDED
#define CALL_GENERIC_H_INCLUDED

#include "mmcore/AbstractGetDataCall.h"
#include "mmcore/BoundingBoxes_2.h"

namespace megamol {
namespace core {

/**
 * Meta data for spatial 3D data communicates the data bounding box as well as frame count
 * and current frame ID for time dependent data.
 */
struct Spatial3DMetaData {
    unsigned int m_frame_cnt = 0;
    unsigned int m_frame_ID = 0;
    megamol::core::BoundingBoxes_2 m_bboxs;
};

enum class RealTimeUnit {
    Unknown,
    Femtosecond,
    Picosecond,
    Nanosecond,
    Microsecond,
    Millisecond,
    Second,
    Minute,
    Hour,
    Day,
    Redshift
};

/** you can request data in two ways and receive an update to show what time span
  * the response actually covers.
  * a) a time step
  * - you request StartTime = EndTime
  * - you will receive, if possible, data with
  *   StartTime_response <= StartTime_request and
  *   EndTime_response >= EndTime_request
  * -> a normal data module will answer the request exactly
  * -> an aggregating module can give you the data for a time span that
  *    includes your request. It depends on the module whether
  *    StartTime_response = StartTime_request or the next best "bin" that is available.
  *    It also depends whether the "bins" are disjunct or a sliding window etc.
  *    You need to look into the response to make sure what you actually get.
  * b) an aggregation
  * - you request StartTime + EndTime
  * -> a normal data module will answer StartTime_response = EndTime_response = StartTime_request
  *    ('I cannot do that')
  * -> an aggregating module must honor this request exactly (if the data exists)
  */
struct TimeParams {
    struct {
        int32_t StartTime = 0;
        int32_t EndTime = 0;
        float RealTimeStart = 0.0f;
        float RealTimeEnd = 0.0f;
        RealTimeUnit Unit = RealTimeUnit::Unknown;
    } DataSet;
    /** this is just a single data "record", we define here what kind
     * of time slice it refers to. Aggregating modules would still return
     * on "time step" but reveal what kind of time span this refers to.
     * Times are always floor, and the fraction is added on top of that,
     * e.g. a module interpolating between discrete simulation time steps
     * (numbered as ints, specifically 0 and 1)
     * [0.8 - 0.8] -> StartTime = 0; StartFraction = 0.8; EndTime = 0; EndFraction = 0.8;
     * e.g. a module aggregating several time steps 5,6,7,8,9,10
     * [5-10] -> StartTime = 5; StartFraction = 0.0; EndTime = 10; EndFraction = 0.0;
     *
     * Note that a response could not fit your request. There are several scenarios for this:
     * a) you request an interpolated frame that cannot be answered:
     *   [1.3] -> StartTime = 1; StartFraction = 0.0;
     * b) you request an interpolated frame and there is a numerical problem:
     *   [1.3] -> StartTime = 1; StartFraction = 0.29999999;
     * c) you request an integral frame and it's still being loaded:
     *   [5] -> StartTime = 2; StartFraction = 0.0;
     * In cases a) and b) HasUpdate considers both time and fraction. In both cases re-requesting
     * will never improve/change the answer, so HasUpdate will only trigger the first time around.
     * In case c) HasUpdate also considers both time and fraction. Re-requesting is useful since
     * the streaming loader will at some time deliver the requested frame.
     * The "ForceFrame" scenario works similar to before: you have to re-issue the call until you
     * get a StartTime that fits your request. HasUpdate can only shortcut complex checks if nothing
     * at all has happened.
     */
    struct {
        int32_t StartTime = 0;
        float StartFraction = 0.0f; // optional, required for interpolation
        int32_t EndTime = 0;
        float EndFraction = 0.0f; // optional, required for interpolation
        float RealTimeStart = 0.0f; // for time+fraction. this CANNOT be requested, just replied.
        float RealTimeEnd = 0.0f; // for time+fraction. this CANNOT be requested, just replied.
        RealTimeUnit Unit = RealTimeUnit::Unknown;
    } Slice;
};

struct EmptyMetaData {
};

template <typename DataType, typename MetaDataType> class GenericVersionedCall : public Call {
public:
    using data_type = DataType;
    using meta_data_type = MetaDataType;

    GenericVersionedCall() = default;
    ~GenericVersionedCall() = default;

    static unsigned int FunctionCount() { return 2; }

    static const unsigned int CallGetData = 0;

    static const unsigned int CallGetMetaData = 1;

    static const char* FunctionName(unsigned int idx) {
        switch (idx) {
        case 0:
            return "GetData";
        case 1:
            return "GetMetaData";
        }
        return NULL;
    }

    void setData(DataType const& data, uint32_t version) {
        m_data = data;
        m_set_version = version;
    }

    void setMetaData(MetaDataType const& meta_data) { m_meta_data = meta_data; }

    DataType const& getData() {
        m_get_version = m_set_version;
        return m_data;
    }

    MetaDataType const& getMetaData() { return m_meta_data; }

    //TODO move setters?

    uint32_t version() { return m_set_version; }

    bool hasUpdate() { return (m_set_version > m_get_version); }

private:
    DataType m_data;
    MetaDataType m_meta_data;

    uint32_t m_get_version = 0;
    uint32_t m_set_version = 0;
};


template<typename DataType, typename MetaDataType>
class GenericTimedVersionedCall : public Call {
public:
    using data_type = DataType;
    using meta_data_type = MetaDataType;

    GenericTimedVersionedCall() = default;
    ~GenericTimedVersionedCall() = default;

    static unsigned int FunctionCount() {
        return 2;
    }

    static const unsigned int CallGetData = 0;

    static const unsigned int CallGetMetaData = 1;

    static const char* FunctionName(unsigned int idx) {
        switch (idx) {
        case 0:
            return "GetData";
        case 1:
            return "GetMetaData";
        }
        return NULL;
    }

    // For compiler reasons I couldn't set a default 0 for frame_id, forcing calls for CallGetMetaData
    // to supply some value, even if the function isn't supposed to be know about a specific frame id yet.
    // In a better world, data and meta data maybe shouldn't be called via the same function, if the parameters
    // aren't even the same....
    bool operator()(unsigned int func, uint32_t frame_id) {
        if (func == CallGetData) {
            m_req_frame_id = frame_id;
        }
        return Call::operator()(func);
    }

    void setData(DataType const& data, uint32_t version, uint32_t frame_id) {
        m_data = data;
        m_set_version = version;
        m_set_frame_id = frame_id;
    }

    void setMetaData(MetaDataType const& meta_data) {
        m_meta_data = meta_data;
    }

    DataType const& getData() {
        m_get_version = m_set_version;
        m_get_frame_id = m_set_frame_id;
        return m_data;
    }

    MetaDataType const& getMetaData() {
        return m_meta_data;
    }

    // TODO move setters?

    uint32_t version() {
        return m_set_version;
    }

    /**
     * Returns frame id of currently set data
     */
    uint32_t frameID() {
        return m_set_frame_id;
    }

    /**
     * Returns frame id from last issued request for data, i.e., from last call to CallData callback
     */
    uint32_t requestedFrameID() {
        return m_req_frame_id;
    }

    bool hasUpdate() {
        bool version_check = (m_set_version > m_get_version);
        // the best I could come up with for a somewhat sane float comparison...see http://realtimecollisiondetection.net/blog/?p=89
        bool frame_id_check = std::abs(m_set_frame_id - m_get_frame_id) >=
                              (std::numeric_limits<float>::epsilon() *
                                  std::max(1.0f, std::max(std::abs(m_set_frame_id), std::abs(m_get_frame_id))));
        return (version_check || frame_id_check);
    }

private:

    /**
    * "Block" original callback operator to users of GenericTimedVersionedCall
    */
    virtual bool operator()(unsigned int func = 0) override final {
        return false;
    }

    DataType m_data;
    MetaDataType m_meta_data;

    uint32_t m_get_version = 0;
    uint32_t m_set_version = 0;

    // comment: still not too thrilled about the name "frame id" because of the ambiguity of
    // simulation frame and render frame (and I usually refer to the latter whenever I use just "frame")
    float m_get_frame_id = 0;
    float m_set_frame_id = 0;
    // comment: technically we could abuse m_set_frame_id to handle requested frame ids but
    // if a module dares not to call setData with the proper frame id during a callback
    // things would probably break, therefore: using a dedicated variable for requested frame id
    float m_req_frame_id = 0;
};

} // namespace mesh
} // namespace megamol

#endif // !CALL_GENERIC_H_INCLUDED
