#include "../include/mmcore/EventStorage.h"

#include "mmcore/EventCall.h"

megamol::core::EventStorage::EventStorage()
    : readEventsSlot("readEvents", "Provides read-only access to pending events to clients.")
    , writeEventsSlot("writeEvents", "Provides write access to event collection to clients.") 
    , m_events({std::make_shared<EventCollection>(), std::make_shared<EventCollection>()})
    , m_read_idx(0) {

    this->readEventsSlot.SetCallback(EventCallRead::ClassName(),
        EventCallRead::FunctionName(EventCallRead::CallGetData), &EventStorage::readDataCallback);
    this->readEventsSlot.SetCallback(EventCallRead::ClassName(),
        EventCallRead::FunctionName(EventCallRead::CallGetMetaData), &EventStorage::readMetaDataCallback);
    this->MakeSlotAvailable(&this->readEventsSlot);

    this->writeEventsSlot.SetCallback(EventCallWrite::ClassName(),
        EventCallWrite::FunctionName(EventCallWrite::CallGetData), &EventStorage::writeDataCallback);
    this->writeEventsSlot.SetCallback(EventCallWrite::ClassName(),
        EventCallWrite::FunctionName(EventCallWrite::CallGetMetaData), &EventStorage::writeMetaDataCallback);
    this->MakeSlotAvailable(&this->writeEventsSlot);
}

megamol::core::EventStorage::~EventStorage() { this->Release(); }

bool megamol::core::EventStorage::create(void) { return true; }

void megamol::core::EventStorage::release(void) {}

bool megamol::core::EventStorage::readDataCallback(core::Call& caller) { 
    auto ec = dynamic_cast<EventCallRead*>(&caller);
    if (ec == nullptr) return false;

    ec->setData(m_events[m_read_idx], this->version);
    
    return true;
}

bool megamol::core::EventStorage::writeDataCallback(core::Call& caller) { 
    auto ec = dynamic_cast<EventCallRead*>(&caller);
    if (ec == nullptr) return false;

    if (ec->version() > this->version) {
        this->version = ec->version();
        auto write_idx = m_read_idx;
        m_read_idx = m_read_idx == 0 ? 1 : 0;
        m_events[write_idx]->clear();

        ec->setData(m_events[write_idx], this->version);
        
    } else if (ec->version() == this->version) {
        auto write_idx = m_read_idx == 0 ? 1 : 0;
        ec->setData(m_events[write_idx], this->version);
    } else {
        return false;
    }

    return true;
}

bool megamol::core::EventStorage::readMetaDataCallback(core::Call& caller) { return true; }

bool megamol::core::EventStorage::writeMetaDataCallback(core::Call& caller) { return true; }
