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
#include <type_traits>
#include <typeindex>
#include <unordered_map>
#include <vector>

namespace megamol {
namespace core {

class EventCollection {
private:
    struct BaseEvent {
        BaseEvent(size_t frame_id) : frame_id(frame_id) {}

        size_t frame_id;
    };

public:
    template<bool is_consumable>
    struct Event : public BaseEvent {
        using is_consumable_t = typename std::conditional<is_consumable, std::true_type, std::false_type>::type;

        Event(size_t frame_id) : BaseEvent(frame_id) {}
    };


    EventCollection() = default;
    ~EventCollection() = default;

    /**
     *
     */
    template<typename EventType>
    std::vector<EventType> get() const;

    /**
     *
     */
    template<typename EventType>
    void add(std::unique_ptr<EventType>&& event);

    /**
     *
     */
    template<typename EventType>
    void remove();

    /**
     *
     */
    void clear();

private:
    // Note to future maintainer: Use of pointer type for individual events is part of
    // what turns this into a generic solution that requires no knowledge about the actual
    // event types contained later on. At the same time, it will propably not scale with
    // larger amounts of events
    std::unordered_multimap<std::type_index, std::unique_ptr<BaseEvent>> m_events;

    // template <class EventType> inline static int getTypeId() {
    //    static const int id = last_type_id++;
    //    return id;
    //}
    //
    // static std::atomic_int last_type_id;
};

template<typename EventType>
inline std::vector<EventType> EventCollection::get() const {

    std::vector<EventType> retval;

    auto range = m_events.equal_range(std::type_index(typeid(EventType)));
    for (auto it = range.first; it != range.second; ++it) {
        retval.push_back(*(static_cast<EventType*>(it->second.get())));
    }

    return retval;
}

template<typename EventType>
inline void EventCollection::add(std::unique_ptr<EventType>&& event) {

    m_events.emplace(std::type_index(typeid(EventType)), std::forward<std::unique_ptr<EventType>>(event));
}

template<typename EventType>
inline void EventCollection::remove() {

    m_events.erase(std::type_index(typeid(EventType)));
}

inline void EventCollection::clear() {
    m_events.clear();
}

} // namespace core
} // namespace megamol


#endif // !MEGAMOL_EVENT_COLLECTION_H_INCLUDED
