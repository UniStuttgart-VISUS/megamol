#include "EventStorage.h"

#include "FrameStatistics.h"
#include "mmstd/event/EventCall.h"

megamol::core::EventStorage::EventStorage()
        : m_events_slot("deployEventCollection", "Provides read-only access to pending events to clients.")
        , m_events(std::make_shared<DoubleBufferedEventCollection>())
        , m_version(0) {

    this->m_events_slot.SetCallback(
        CallEvent::ClassName(), CallEvent::FunctionName(CallEvent::CallGetData), &EventStorage::dataCallback);
    this->m_events_slot.SetCallback(
        CallEvent::ClassName(), CallEvent::FunctionName(CallEvent::CallGetMetaData), &EventStorage::metaDataCallback);
    this->MakeSlotAvailable(&this->m_events_slot);
}

megamol::core::EventStorage::~EventStorage() {
    this->Release();
}

bool megamol::core::EventStorage::create(void) {
    return true;
}

void megamol::core::EventStorage::release(void) {}

bool megamol::core::EventStorage::dataCallback(core::Call& caller) {
    auto ec = dynamic_cast<CallEvent*>(&caller);
    if (ec == nullptr)
        return false;

    auto current_frame_id = frontend_resources.get<frontend_resources::FrameStatistics>().rendered_frames_count;

    // the first call in a frame to either readData or writeData needs to swap the double buffer
    if (current_frame_id > this->m_version) {
        this->m_version = current_frame_id;
        m_events->swap();
    }

    ec->setData(m_events, current_frame_id);

    return true;
}

bool megamol::core::EventStorage::metaDataCallback(core::Call& caller) {
    return true;
}
