#include <iostream>
#include <iomanip>

#include "OmniUsdReader.h"
#define TBB_USE_DEBUG 0
#include "pxr/usd/usd/stage.h"
#include "OmniClient.h"

#include "mmcore/param/FilePathParam.h"

megamol::mesh::OmniUsdReader::OmniUsdReader()
        : AbstractMeshDataSource()
        , m_filename_slot("Omniverse URL to USD stage", "URL to omniverseserver") {
    this->m_filename_slot << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->m_filename_slot);
}

megamol::mesh::OmniUsdReader::~OmniUsdReader() {}

static void OmniClientConnectionStatusCallbackImpl(
    void* userData, const char* url, OmniClientConnectionStatus status) noexcept {
    std::cout << "Connection Status: " << omniClientGetConnectionStatusString(status) << " [" << url << "]"
              << std::endl;
    if (status == eOmniClientConnectionStatus_ConnectError) {
        std::cout << "[ERROR] Failed connection, exiting." << std::endl;
        exit(-1);
    }
}

// Startup Omniverse
static bool startOmniverse() {
    // Register a function to be called whenever the library wants to print something to a log
    omniClientSetLogCallback(
        [](char const* threadName, char const* component, OmniClientLogLevel level, char const* message) {
            std::cout << "[" << omniClientGetLogLevelString(level) << "] " << message << std::endl;
        });

    // The default log level is "Info", set it to "Debug" to see all messages
    omniClientSetLogLevel(eOmniClientLogLevel_Info);

    // Initialize the library and pass it the version constant defined in OmniClient.h
    // This allows the library to verify it was built with a compatible version. It will
    // return false if there is a version mismatch.
    if (!omniClientInitialize(kOmniClientVersion)) {
        return false;
    }

    omniClientRegisterConnectionStatusCallback(nullptr, OmniClientConnectionStatusCallbackImpl);
    return true;
}



bool megamol::mesh::OmniUsdReader::create(void) {
    AbstractMeshDataSource::create();
    startOmniverse();
    return true;
}

bool megamol::mesh::OmniUsdReader::getMeshDataCallback(core::Call& caller) {

    return true;
}

bool megamol::mesh::OmniUsdReader::getMeshMetaDataCallback(core::Call& caller) {
    return AbstractMeshDataSource::getMeshMetaDataCallback(caller);
}

void megamol::mesh::OmniUsdReader::release() {}

