/*
 * EventCollection.h
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_EVENT_COLLECTION_H_INCLUDED
#define MEGAMOL_EVENT_COLLECTION_H_INCLUDED

#include <atomic>
#include <memory>
#include <vector>
#include <unordered_map>

namespace megamol {
namespace core {

class EventCollection {
public:

    class AbstractEvent {
    public:
        AbstractEvent() = default;
        ~AbstractEvent() = default;

        size_t frame_id;
    };

    EventCollection() = default;
    ~EventCollection() = default;

    /**
     *
     */
    template <typename EventType>
    std::vector<EventType> get() const;

    /**
     *
     */
    template <typename EventType> 
    void add(std::unique_ptr<AbstractEvent>&& event);

    void clear();

private:

    // Note to future maintainer: Use of pointer type for individual events is part of
    // what turns this into a generic solution that requires no knowledge about the actual
    // event types contained later on. At the same time, it will propably not scale with
    // larger amounts of events
    std::unordered_multimap<int, std::unique_ptr<AbstractEvent>> m_events;

    template <class EventType> inline static int getTypeId() {
        static const int id = last_type_id++;
        return id;
    }

    static std::atomic_int last_type_id;
};

template <typename EventType>
inline std::vector<EventType> EventCollection::get() const {

    std::vector<EventType> retval;

    auto range = m_events.find(getTypeId<EventType>());
    for (auto it = range.first; it != range.second; ++it) {
        retval.push_back(*(static_cast<EventType*>(it->second.get())));
    }

    return retval;
}

template <typename EventType>
inline void EventCollection::add(std::unique_ptr<AbstractEvent>&& event) {

    m_events.emplace(getTypeId<EventType>(), std::forward<std::unique_ptr<AbstractEvent>>(event));
}

inline void EventCollection::clear() { m_events.clear(); }

} // namespace megamol
}


#endif // !MEGAMOL_EVENT_COLLECTION_H_INCLUDED
