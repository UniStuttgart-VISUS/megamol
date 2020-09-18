#include "../include/mmcore/EventStorage.h"

#include "mmcore/CoreInstance.h"
#include "mmcore/EventCall.h"

megamol::core::EventStorage::EventStorage()
    : m_readEventsSlot("readEvents", "Provides read-only access to pending events to clients.")
    , m_writeEventsSlot("writeEvents", "Provides write access to event collection to clients.") 
    , m_events({std::make_shared<EventCollection>(), std::make_shared<EventCollection>()})
    , m_read_idx(0)
    , m_version(0) {

    this->m_readEventsSlot.SetCallback(EventCallRead::ClassName(),
        EventCallRead::FunctionName(EventCallRead::CallGetData), &EventStorage::readDataCallback);
    this->m_readEventsSlot.SetCallback(EventCallRead::ClassName(),
        EventCallRead::FunctionName(EventCallRead::CallGetMetaData), &EventStorage::readMetaDataCallback);
    this->MakeSlotAvailable(&this->m_readEventsSlot);

    this->m_writeEventsSlot.SetCallback(EventCallWrite::ClassName(),
        EventCallWrite::FunctionName(EventCallWrite::CallGetData), &EventStorage::writeDataCallback);
    this->m_writeEventsSlot.SetCallback(EventCallWrite::ClassName(),
        EventCallWrite::FunctionName(EventCallWrite::CallGetMetaData), &EventStorage::writeMetaDataCallback);
    this->MakeSlotAvailable(&this->m_writeEventsSlot);
}

megamol::core::EventStorage::~EventStorage() { this->Release(); }

bool megamol::core::EventStorage::create(void) { return true; }

void megamol::core::EventStorage::release(void) {}

bool megamol::core::EventStorage::readDataCallback(core::Call& caller) { 
    auto ec = dynamic_cast<EventCallRead*>(&caller);
    if (ec == nullptr) return false;

    ec->setData(m_events[m_read_idx], this->GetCoreInstance()->GetFrameID());
    
    return true;
}

bool megamol::core::EventStorage::writeDataCallback(core::Call& caller) { 
    auto ec = dynamic_cast<EventCallRead*>(&caller);
    if (ec == nullptr) return false;

    auto current_frame_id = this->GetCoreInstance()->GetFrameID();

    if (current_frame_id > this->m_version) {
        this->m_version = current_frame_id;
        auto write_idx = m_read_idx;
        m_read_idx = m_read_idx == 0 ? 1 : 0;
        m_events[write_idx]->clear();

        ec->setData(m_events[write_idx], current_frame_id);
        
    } else {
        auto write_idx = m_read_idx == 0 ? 1 : 0;
        ec->setData(m_events[write_idx], current_frame_id);
    }

    return true;
}

bool megamol::core::EventStorage::readMetaDataCallback(core::Call& caller) { return true; }

bool megamol::core::EventStorage::writeMetaDataCallback(core::Call& caller) { return true; }
