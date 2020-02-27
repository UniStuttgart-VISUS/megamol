/*
 * Graph.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_STOCK_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_STOCK_H_INCLUDED


#include <map>
#include <vector>


namespace megamol {
namespace gui {
namespace configurator {


// GRAPH STOCK DATA STRUCTURE ---------------------------------------------

struct StockParameter {
    std::string class_name;
    std::string description;
    Parameter::ParamType type;
    std::string value_string;
};

struct StockCallSlot {
    std::string name;
    std::string description;
    std::vector<size_t> compatible_call_idxs;
    CallSlot::CallSlotType type;
};

struct StockCall {
    std::string class_name;
    std::string description;
    std::string plugin_name;
    std::vector<std::string> functions;
};

struct StockModule {
    std::string class_name;
    std::string description;
    std::string plugin_name;
    bool is_view;
    std::vector<StockParameter> parameters;
    std::map<CallSlot::CallSlotType, std::vector<StockCallSlot>> call_slots;
};

typedef std::vector<StockModule> ModuleStockType;
typedef std::vector<StockCall> CallStockType;


} // namespace configurator
} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_STOCK_H_INCLUDED