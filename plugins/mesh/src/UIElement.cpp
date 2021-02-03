#include "stdafx.h"

#include "UIElement.h"

megamol::mesh::UIElement::UIElement() 
    : core::Module() 
    , m_getData_slot("getUIElement", "Slot publishing the module's UI element")
    , m_UIElement_callerSlot("chainUIElement", "Slot for chaining UI element modules") 
{
    this->m_getData_slot.SetCallback(Call3DInteraction::ClassName(), "GetData", &UIElement::getDataCallback);
    this->m_getData_slot.SetCallback(Call3DInteraction::ClassName(), "GetMetaData", &UIElement::getMetaDataCallback);
    this->MakeSlotAvailable(&this->m_getData_slot);

    this->m_UIElement_callerSlot.SetCompatibleCall<Call3DInteractionDescription>();
    this->MakeSlotAvailable(&this->m_UIElement_callerSlot);
}

megamol::mesh::UIElement::~UIElement() {}

bool megamol::mesh::UIElement::create(void) {
    return true; 
}

bool megamol::mesh::UIElement::getDataCallback(core::Call& caller) { 
    return true; 
}

bool megamol::mesh::UIElement::getMetaDataCallback(core::Call& caller) { 
    return true; 
}

void megamol::mesh::UIElement::release() {}
