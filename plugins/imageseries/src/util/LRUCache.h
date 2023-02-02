#pragma once

/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#include <memory>
#include <unordered_map>

namespace megamol::ImageSeries::util {

template<typename Key, typename Value>
class LRUCache {
public:
    using SizeGetterFunc = std::function<std::size_t(const Value&)>;

    LRUCache(SizeGetterFunc sizeGetter) : sizeGetter(sizeGetter) {}

    void clear() {
        entries.clear();
        accessCount = 0;
        totalByteCount = 0;
    }

    std::shared_ptr<const Value> get(const Key& key) const {
        auto result = entries.find(key);
        if (result != entries.end()) {
            result->second.lastAccess = accessCount++;
            return result->second.value;
        } else {
            return nullptr;
        }
    }

    std::shared_ptr<const Value> operator[](const Key& key) const {
        return get(key);
    }

    template<typename Func>
    std::shared_ptr<const Value> findOrCreate(const Key& key, Func supplier) {
        if (maximumSize == 0) {
            return supplier(key);
        }

        auto result = entries.find(key);
        if (result != entries.end()) {
            result->second.lastAccess = accessCount++;
            return result->second.value;
        } else {
            Entry entry;
            entry.lastAccess = accessCount++;
            entry.value = supplier(key);
            entry.byteCount = entry.value != nullptr ? sizeGetter(*entry.value) : 0;

            totalByteCount += entry.byteCount;
            entries.insert(std::make_pair(key, entry));

            cleanUp();

            return entry.value;
        }
    }

    std::shared_ptr<const Value> find(const Key& key) {
        if (maximumSize == 0) {
            throw std::runtime_error("Find function is not allowed if cache size is zero.");
        }

        auto result = entries.find(key);
        if (result != entries.end()) {
            result->second.lastAccess = accessCount++;
            return result->second.value;
        } else {
            return nullptr;
        }
    }

    void setMaximumSize(std::size_t maximumSize) {
        if (this->maximumSize != maximumSize) {
            this->maximumSize = maximumSize;
            cleanUp();
        }
    }

    std::size_t getMaximumSize() const {
        return maximumSize;
    }

private:
    struct Entry {
        std::shared_ptr<const Value> value;
        mutable std::size_t lastAccess = 0;
        std::size_t byteCount = 0;
    };

    void cleanUp() {
        if (maximumSize == 0) {
            entries.clear();
        } else if (totalByteCount > maximumSize) {
            // Obtain list of entries
            std::vector<std::pair<Key, Entry>> entryList;
            for (auto it = entries.begin(); it != entries.end(); ++it) {
                entryList.emplace_back(it->first, it->second);
            }

            // Sort by last access time (oldest entries last)
            std::sort(entryList.begin(), entryList.end(),
                [](const auto& a, const auto& b) { return a.second.lastAccess > b.second.lastAccess; });

            // Clean up until total memory usage is below threshold
            std::size_t cleanupThreshold = maximumSize * cleanupFactor;
            while (totalByteCount > cleanupThreshold && !entryList.empty()) {
                totalByteCount -= entryList.back().second.byteCount;
                entries.erase(entryList.back().first);
                entryList.pop_back();
            }
        }
    }

    SizeGetterFunc sizeGetter;

    std::unordered_map<Key, Entry> entries;
    mutable std::size_t accessCount = 0;
    std::size_t totalByteCount = 0;
    std::size_t maximumSize = 0;
    float cleanupFactor = 0.9;
};

} // namespace megamol::ImageSeries::util
