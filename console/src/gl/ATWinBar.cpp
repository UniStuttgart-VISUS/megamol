/*
 * gl/ATWinBar.cpp
 *
 * Copyright (C) 2016 MegaMol Team
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#ifdef HAS_ANTTWEAKBAR
#include "gl/Window.h"
#include "gl/ATWinBar.h"
#include <sstream>
#include "GLFW/glfw3.h"
#include "gl/ATBUILayer.h"
#include "utility/ParamFileManager.h"

using namespace megamol;
using namespace megamol::console;

gl::ATWinBar::ATWinBar(ATBUILayer& layer, const char* wndName) : ATBar("winBar"), layer(layer),
        winX(0), winY(0), winW(0), winH(0), winSizePresets() {

    winSizePresets[0] = std::tuple<ATWinBar*, unsigned int, unsigned int>(this, 256, 256);
    winSizePresets[1] = std::tuple<ATWinBar*, unsigned int, unsigned int>(this, 512, 512);
    winSizePresets[2] = std::tuple<ATWinBar*, unsigned int, unsigned int>(this, 1024, 1024);
    winSizePresets[3] = std::tuple<ATWinBar*, unsigned int, unsigned int>(this, 1280, 720);
    winSizePresets[4] = std::tuple<ATWinBar*, unsigned int, unsigned int>(this, 1920, 1080);

    std::stringstream def;
    def << Name() << " "
        << "label='MegaMol™ - " << wndName << "' "
        << "help='Options regarding the Window and MegaMol Instance' "
        << "color='225 230 235' alpha='192' text='dark' "
        << "position='48 48' "
        << "size='256 256' valueswidth='128' ";
    ::TwDefine(def.str().c_str());

    // FPS
    ::TwAddVarRO(Handle(), "fps", TW_TYPE_FLOAT, &this->fps, "label='FPS' help='Rendering performance in frames per second, measured by the time between OpenGL buffer swaps. This value is the mean over several rendering calls or over the rendering calls within some time'");
    ::TwAddVarCB(Handle(), "fpsTitle", TW_TYPE_BOOLCPP, 
        reinterpret_cast<TwSetVarCallback>(&ATWinBar::setShowFPSinWindowTitle),
        reinterpret_cast<TwGetVarCallback>(&ATWinBar::getShowFPSinWindowTitle),
        nullptr, "label='Show FPS in Window Caption'");
    ::TwAddVarCB(Handle(), "sampTitle", TW_TYPE_BOOLCPP,
        reinterpret_cast<TwSetVarCallback>(&ATWinBar::setShowSamplesinWindowTitle),
        reinterpret_cast<TwGetVarCallback>(&ATWinBar::getShowSamplesinWindowTitle),
        nullptr, "label='Show Samples passed in Window Caption'");
    ::TwAddVarCB(Handle(), "primsTitle", TW_TYPE_BOOLCPP,
        reinterpret_cast<TwSetVarCallback>(&ATWinBar::setShowPrimsinWindowTitle),
        reinterpret_cast<TwGetVarCallback>(&ATWinBar::getShowPrimsinWindowTitle),
        nullptr, "label='Show Primitives generated in Window Caption'");
    ::TwAddButton(Handle(), "fpsCopy", reinterpret_cast<TwButtonCallback>(&ATWinBar::copyFPS), nullptr, "label='Copy to Clipboard'");
    ::TwAddButton(Handle(), "fpsCopyList", reinterpret_cast<TwButtonCallback>(&ATWinBar::copyFPSList), nullptr, "label='Copy FPS List to Clipboard'");

    // Window placement, size, decorations, etc.
    //  Note: switching to/from fullscreen mode is no longer supported, as GLFW cannot remove/add window decorations after window was created.
    ::TwAddVarRW(Handle(), "winX", TW_TYPE_INT32, &winX, "label='Left' group='Window'");
    ::TwAddVarRW(Handle(), "winY", TW_TYPE_INT32, &winY, "label='Top' group='Window'");
    ::TwAddVarRW(Handle(), "winW", TW_TYPE_UINT32, &winW, "label='Width' group='Window'");
    ::TwAddVarRW(Handle(), "winH", TW_TYPE_UINT32, &winH, "label='Height' group='Window'");
    ::TwAddSeparator(Handle(), nullptr, "group='Window'");
    getWndValues(this);

    ::TwAddButton(Handle(), "winValGet", reinterpret_cast<TwButtonCallback>(&ATWinBar::getWndValues), this, "label='Get Values' group='Window'");
    ::TwAddButton(Handle(), "winValSet", reinterpret_cast<TwButtonCallback>(&ATWinBar::setWndValues), this, "label='Set Values' group='Window'");

    unsigned int i = 0;
    for (auto& wsp : winSizePresets) {
        def.str("");
        def << "winSizePreset" << i;
        std::string name = def.str();
        def.str("");
        def << "label='Set Size " << std::get<1>(wsp) << "x" << std::get<2>(wsp) << "' group='WinSizePre'";
        ::TwAddButton(Handle(), name.c_str(), reinterpret_cast<TwButtonCallback>(&ATWinBar::setWinSizePreset), &wsp, def.str().c_str());
        ++i;
    }
    def.str("");
    def << Name() << "/WinSizePre "
        << "label='Size Presets' "
        << "group='Window' "
        << "opened=true ";
    ::TwDefine(def.str().c_str());
    def.str("");
    def << Name() << "/Window "
        << "opened=false ";
    ::TwDefine(def.str().c_str());

    // GUI show/hide control
    ::TwAddButton(Handle(), "hideGUI", reinterpret_cast<TwButtonCallback>(&ATWinBar::toggleGUI), this, "label='Hide/Show GUI' key=F12");

    // ParamFile
    ::TwAddVarCB(Handle(), "paramFilePath", TW_TYPE_CDSTRING, 
        reinterpret_cast<TwSetVarCallback>(&ATWinBar::setParamFilePath),
        reinterpret_cast<TwGetVarCallback>(&ATWinBar::getParamFilePath),
        nullptr, "label='ParamFile' group='paramFile'");
    ::TwAddButton(Handle(), "loadParamFile", reinterpret_cast<TwButtonCallback>(&ATWinBar::loadParamFile), nullptr, "label='Load ParamFile' group='paramFile'");
    ::TwAddButton(Handle(), "saveParamFile", reinterpret_cast<TwButtonCallback>(&ATWinBar::saveParamFile), nullptr, "label='Save ParamFile' group='paramFile'");
    def.str("");
    def << Name() << "/paramFile "
        << "label='ParamFile' "
        << "opened=true ";
    ::TwDefine(def.str().c_str());

    // (later: ProjectFile)

    def.str("");
    def << Name() << " "
        << "iconified=true ";
    ::TwDefine(def.str().c_str());

}

gl::ATWinBar::~ATWinBar() {
}

void gl::ATWinBar::getWndValues(ATWinBar *inst) {
//    ::glfwGetWindowPos(inst->wnd.WindowHandle(), &inst->winX, &inst->winY);
//    int w, h;
//    ::glfwGetFramebufferSize(inst->wnd.WindowHandle(), &w, &h);
//    if (w < 0) w = 0;
//    if (h < 0) h = 0;
//    inst->winW = static_cast<unsigned int>(w);
//    inst->winH = static_cast<unsigned int>(h);
}

void gl::ATWinBar::setWndValues(ATWinBar *inst) {
//    int x, y;
//    ::glfwGetFramebufferSize(inst->wnd.WindowHandle(), &x, &y);
//    if (x < 0) x = 0;
//    if (y < 0) y = 0;
//    if ((static_cast<unsigned int>(x) != inst->winW) || (static_cast<unsigned int>(y) != inst->winH)) {
//        ::glfwSetWindowSize(inst->wnd.WindowHandle(), inst->winW, inst->winH);
//    }
//    ::glfwGetWindowPos(inst->wnd.WindowHandle(), &x, &y);
//    if ((x != inst->winX) || (y != inst->winY)) {
//        ::glfwSetWindowPos(inst->wnd.WindowHandle(), inst->winX, inst->winY);
//    }
//    getWndValues(inst);
}

void gl::ATWinBar::setWinSizePreset(std::tuple<ATWinBar*, unsigned int, unsigned int> *preset) {
//    ::glfwSetWindowSize(std::get<0>(*preset)->wnd.WindowHandle(), std::get<1>(*preset), std::get<2>(*preset));
//    getWndValues(std::get<0>(*preset));
}

void gl::ATWinBar::toggleGUI(ATWinBar *inst) {
    inst->layer.ToggleEnable();
}

void gl::ATWinBar::setShowFPSinWindowTitle(const void *value, Window *wnd) {
    //wnd->SetShowFPSinTitle(*(const bool*)value);
}

void gl::ATWinBar::getShowFPSinWindowTitle(void *value, Window *wnd) {
    //*(bool*)value = wnd->ShowFPSinTitle();
}

void gl::ATWinBar::setShowSamplesinWindowTitle(const void *value, Window *wnd) {
    //wnd->SetShowSamplesinTitle(*(const bool*)value);
}

void gl::ATWinBar::getShowSamplesinWindowTitle(void *value, Window *wnd) {
    //*(bool*)value = wnd->ShowSamplesinTitle();
}

void gl::ATWinBar::setShowPrimsinWindowTitle(const void *value, Window *wnd) {
    //wnd->SetShowPrimsinTitle(*(const bool*)value);
}

void gl::ATWinBar::getShowPrimsinWindowTitle(void *value, Window *wnd) {
    //*(bool*)value = wnd->ShowPrimsinTitle();
}

void gl::ATWinBar::copyFPS(Window *wnd) {
    std::stringstream str;
    //str << wnd->LiveFPS();
    //::glfwSetClipboardString(wnd->WindowHandle(), str.str().c_str());
}

void gl::ATWinBar::copyFPSList(Window *wnd) {
    std::stringstream str;
    //for (float v : wnd->LiveFPSList()) {
        //str << v << std::endl;
    //}
    //::glfwSetClipboardString(wnd->WindowHandle(), str.str().c_str());
}

void gl::ATWinBar::setParamFilePath(const void *value, void *ctxt) {
    // Set: copy the value from AntTweakBar
    const char *src = *(const char **)(value);
    utility::ParamFileManager::Instance().filename = vislib::StringA(src);
}

void gl::ATWinBar::getParamFilePath(void *value, void *ctxt) {
    // Get: copy the value to AntTweakBar
    char **destPtr = (char **)value;
    TwCopyCDStringToLibrary(destPtr, vislib::StringA(utility::ParamFileManager::Instance().filename).PeekBuffer());
}

void gl::ATWinBar::loadParamFile(void *ctxt) {
    utility::ParamFileManager::Instance().Load();
}

void gl::ATWinBar::saveParamFile(void *ctxt) {
    utility::ParamFileManager::Instance().Save();
}
#endif
