#pragma once

#include <array>
#include <string>
#include <type_traits>

#include "implot.h"
#include "mmcore/MultiPerformanceHistory.h"

namespace megamol::gui {
class ProfilingUtils {
public:
    enum class MetricType { MINMAXAVG = 0, SUM = 1 };
    static inline const std::array<std::string, 2> MetricNames = {"Min/Max+Avg", "Sum"};

    template<std::size_t sz>
    static std::array<core::MultiPerformanceHistory::perf_type, sz> range() {
        std::array<core::MultiPerformanceHistory::perf_type, sz> arr{0.0};
        std::iota(arr.begin(), arr.end(), 0.0);
        return arr;
    }
    inline static auto xbuf = range<core::MultiPerformanceHistory::buffer_length>();

    template<typename E>
    static constexpr auto to_underlying(E e) noexcept {
        return static_cast<std::underlying_type_t<E>>(e);
    }

    class ProxyVector {
    private:
        std::vector<core::MultiPerformanceHistory*> my_histories;

    public:
        ProxyVector() = default;
        void append(
            std::unordered_map<frontend_resources::PerformanceManager::handle_type, core::MultiPerformanceHistory>&
                histories) {
            for (auto& item : histories) {
                my_histories.push_back(&(item.second));
            }
        }
        core::MultiPerformanceHistory& operator[](const size_t& i) const {
            return *my_histories[i];
        }
        const size_t size() const {
            return my_histories.size();
        }
    };

    static void PrintTableRow(const std::string& label, float data) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(label.c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%.12f", data);
    }
    static void PrintTableRow(const std::string& label, int data) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(label.c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%i", data);
    }
    static void MetricDropDown(MetricType& selectedIndex, ImPlotAxisFlags& y_flags) {
        if (ImGui::BeginCombo("Display", MetricNames[to_underlying(selectedIndex)].c_str())) {
            for (auto i = 0; i < MetricNames.size(); ++i) {
                bool isSelected = (i == to_underlying(selectedIndex));
                if (ImGui::Selectable(MetricNames[i].c_str(), isSelected)) {
                    selectedIndex = static_cast<MetricType>(i);
                    y_flags = ImPlotAxisFlags_AutoFit;
                }
            }
            ImGui::EndCombo();
        }
    }
    static void DrawPlot(std::string name, ImVec2 size, const ImPlotAxisFlags& y_flags,
        megamol::gui::ProfilingUtils::MetricType display_idx, const megamol::core::MultiPerformanceHistory& history) {
        if (ImPlot::BeginPlot(name.c_str(), nullptr, "ms", size, ImPlotFlags_None, ImPlotAxisFlags_AutoFit, y_flags)) {
            if (display_idx == ProfilingUtils::MetricType::MINMAXAVG) {
                ImPlot::PlotShaded(("###" + name + "minmax").c_str(), xbuf.data(),
                    history.copyHistory(core::MultiPerformanceHistory::metric_type::MIN).data(),
                    history.copyHistory(core::MultiPerformanceHistory::metric_type::MAX).data(),
                    core::MultiPerformanceHistory::buffer_length);
                ImPlot::PlotLine(("###" + name + "plot").c_str(), xbuf.data(),
                    history.copyHistory(core::MultiPerformanceHistory::metric_type::AVERAGE).data(),
                    core::MultiPerformanceHistory::buffer_length);
            } else {
                ImPlot::PlotLine(("###" + name + "plot").c_str(), xbuf.data(),
                    history.copyHistory(core::MultiPerformanceHistory::metric_type::SUM).data(),
                    core::MultiPerformanceHistory::buffer_length);
            }
            ImPlot::EndPlot();
        }
    }
};
} // namespace megamol::gui
