#include "stdafx.h"

#include "TextureCombine.h"

#include "compositing/CompositingCalls.h"

megamol::compositing::TextureCombine::TextureCombine() 
    : core::Module()
    , m_output_texture(nullptr)
    , m_output_texture_hash(0)
    , m_output_tex_slot("OutputTexture","Gives access to resulting output texture")
    , m_input_tex_0_slot("InputTexture","Connects the primary input texture that is also used the set the output texture size")
    , m_input_tex_1_slot("InputTexture","Connects the secondary input texture") 
{
    this->m_output_tex_slot.SetCallback(
        CallTexture2D::ClassName(), "GetData", &TextureCombine::getDataCallback);
    this->m_output_tex_slot.SetCallback(
        CallTexture2D::ClassName(), "GetMetaData", &TextureCombine::getMetaDataCallback);
    this->MakeSlotAvailable(&this->m_output_tex_slot);

    this->m_input_tex_0_slot.SetCompatibleCall<CallTexture2DDescription>();
    this->MakeSlotAvailable(&this->m_input_tex_0_slot);

    this->m_input_tex_1_slot.SetCompatibleCall<CallTexture2DDescription>();
    this->MakeSlotAvailable(&this->m_input_tex_1_slot);
}

megamol::compositing::TextureCombine::~TextureCombine() { 
    this->Release(); 
}

bool megamol::compositing::TextureCombine::create() {
    return true;
}

void megamol::compositing::TextureCombine::release() {
}

bool megamol::compositing::TextureCombine::getDataCallback(core::Call& caller) 
{ 

    //TODO compute texture combine operation

    return true; 
}

bool megamol::compositing::TextureCombine::getMetaDataCallback(core::Call& caller) {

    //TODO output hash?

    return true;
}
