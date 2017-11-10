#include "stdafx.h"
#include "FBOCompositor.h"

#include <fstream>
#include <queue>

#include "mmcore/CoreInstance.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/utility/ResourceWrapper.h"
#include "mmcore/view/CallRender3D.h"

#include "vislib/sys/Log.h"

#include "FBOTransmitter.h"

using namespace megamol;
using namespace megamol::pbs;


FBOCompositor::FBOCompositor(void) : core::view::Renderer3DModule(),
fboWidthSlot("width", "Sets width of FBO"),
fboHeightSlot("height", "Sets height of FBO"),
ipAddressSlot("address", "IP address of reciever"),
numRenderNodesSlot("numRenderNodes", "Number of render nodes connected to this compositor"),
zmq_ctx(1),
zmq_socket(zmq_ctx, zmq::socket_type::pull),
num_render_nodes(0),
fbo_width(-1),
fbo_height(-1),
color_textures(nullptr),
depth_textures(nullptr) {
    /*this->viewport[0] = 0;
    this->viewport[1] = 0;
    this->viewport[2] = 1;
    this->viewport[3] = 1;*/
    this->fboWidthSlot << new core::param::IntParam(1, 1, 3840);
    this->MakeSlotAvailable(&this->fboWidthSlot);

    this->fboHeightSlot << new core::param::IntParam(1, 1, 2160);
    this->MakeSlotAvailable(&this->fboHeightSlot);

    this->ipAddressSlot << new core::param::StringParam("localhost:34242");
    this->ipAddressSlot.SetUpdateCallback(&FBOCompositor::connectSocketCallback);
    this->MakeSlotAvailable(&this->ipAddressSlot);

    /*this->numRenderNodesSlot << new core::param::IntParam(0, 0, 16);
    this->numRenderNodesSlot.SetUpdateCallback(&FBOCompositor::updateNumRenderNodesCallback);
    this->MakeSlotAvailable(&this->numRenderNodesSlot);*/
}


FBOCompositor::~FBOCompositor(void) {
    this->Release();
}


bool FBOCompositor::printShaderInfoLog(GLuint shader) const {
    int infoLogLen = 0;
    int charsWritten = 0;
    GLchar *infoLog;

    GLint compileStatus;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compileStatus);
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLogLen);

    if (compileStatus == GL_FALSE && infoLogLen > 1) {
        infoLog = new GLchar[infoLogLen];
        glGetShaderInfoLog(shader, infoLogLen, &charsWritten, infoLog);
        fprintf(stderr, "InfoLog : %s\n", infoLog);
        delete[] infoLog;
        return false;
    }
    return (compileStatus == GL_TRUE);
}


bool FBOCompositor::printProgramInfoLog(GLuint shaderProg) const {
    int infoLogLen = 0;
    int charsWritten = 0;
    GLchar *infoLog;

    GLint linkStatus;
    glGetProgramiv(shaderProg, GL_INFO_LOG_LENGTH, &infoLogLen);
    glGetProgramiv(shaderProg, GL_LINK_STATUS, &linkStatus);

    if (linkStatus == GL_FALSE && infoLogLen > 1) {
        infoLog = new GLchar[infoLogLen];
        glGetProgramInfoLog(shaderProg, infoLogLen, &charsWritten, infoLog);
        fprintf(stderr, "\nProgramInfoLog :\n\n%s\n", infoLog);
        delete[] infoLog;
        return false;
    }
    return (linkStatus == GL_TRUE);
}


bool FBOCompositor::create(void) {
    //this->num_render_nodes = this->numRenderNodesSlot.Param<core::param::IntParam>()->Value();

    this->renderData = new std::vector<fbo_data>();
    this->receiverData = new std::vector<fbo_data>();

    glGenVertexArrays(1, &this->vao);
    glBindVertexArray(this->vao);
    glGenBuffers(1, &this->vbo);
    glBindBuffer(GL_ARRAY_BUFFER, this->vbo);
    float buffer[] = {
        -1, -1, 0,
        1, -1, 0,
        1, 1, 0,
        -1, 1, 0
    };
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 12, buffer, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, false, 0, nullptr);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // create shader
    auto vert_shader = glCreateShader(GL_VERTEX_SHADER);
    auto frag_shader = glCreateShader(GL_FRAGMENT_SHADER);

    auto path_to_vert = core::utility::ResourceWrapper::getFileName(this->GetCoreInstance()->Configuration(), "pbscompositor.vert.glsl");
    auto path_to_frag = core::utility::ResourceWrapper::getFileName(this->GetCoreInstance()->Configuration(), "pbscompositor.frag.glsl");

    std::ifstream shader_file(W2A(path_to_vert.PeekBuffer()));
    std::string shader_string = std::string(std::istreambuf_iterator<char>(shader_file), std::istreambuf_iterator<char>());
    shader_file.close();

    auto shader_cstring = shader_string.c_str();
    GLint string_size = shader_string.size();
    glShaderSource(vert_shader, 1, &shader_cstring, &string_size);

    shader_file.open(W2A(path_to_frag.PeekBuffer()));
    shader_string = std::string(std::istreambuf_iterator<char>(shader_file), std::istreambuf_iterator<char>());
    shader_file.close();

    shader_cstring = shader_string.c_str();
    string_size = shader_string.size();
    glShaderSource(frag_shader, 1, &shader_cstring, &string_size);

    glCompileShader(vert_shader);
    if (!printShaderInfoLog(vert_shader)) {
        return false;
    }
    glCompileShader(frag_shader);
    if (!printShaderInfoLog(frag_shader)) {
        return false;
    }

    this->shader = glCreateProgram();
    glAttachShader(this->shader, vert_shader);
    glAttachShader(this->shader, frag_shader);
    glLinkProgram(this->shader);

    if (!printProgramInfoLog(this->shader)) {
        return false;
    }

    glDeleteShader(vert_shader);
    glDeleteShader(frag_shader);

    return true;
}


void FBOCompositor::release(void) {
    glDeleteTextures(this->num_render_nodes, this->color_textures);
    glDeleteTextures(this->num_render_nodes, this->depth_textures);

    ARY_SAFE_DELETE(this->color_textures);
    ARY_SAFE_DELETE(this->depth_textures);

    glDeleteProgram(this->shader);

    glDeleteVertexArrays(1, &this->vao);
    glDeleteBuffers(1, &this->vbo);

    for (int i = 0; i < this->ip_address.size(); i++) {
        this->zmq_socket.disconnect("tcp://" + this->ip_address[i]);
    }

    SAFE_DELETE(this->receiverData);
    SAFE_DELETE(this->renderData);
}


bool FBOCompositor::GetCapabilities(core::Call &call) {
    core::view::CallRender3D *cr = dynamic_cast<core::view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    cr->SetCapabilities(
        core::view::CallRender3D::CAP_RENDER
        | core::view::CallRender3D::CAP_LIGHTING
        | core::view::CallRender3D::CAP_ANIMATION
    );

    return true;
}


bool FBOCompositor::GetExtents(core::Call &call) {
    core::view::CallRender3D *cr = dynamic_cast<core::view::CallRender3D*>(&call);
    if (cr == nullptr) return false;

    cr->SetTimeFramesCount(1);

    auto &out_bbox = cr->AccessBoundingBoxes();

    if (this->num_render_nodes > 0) {
        out_bbox.SetObjectSpaceBBox((*this->renderData)[0].bbox[vislib::math::Cuboid<float>::FACE_LEFT],
            (*this->renderData)[0].bbox[vislib::math::Cuboid<float>::FACE_BOTTOM],
            (*this->renderData)[0].bbox[vislib::math::Cuboid<float>::FACE_BACK],
            (*this->renderData)[0].bbox[vislib::math::Cuboid<float>::FACE_RIGHT],
            (*this->renderData)[0].bbox[vislib::math::Cuboid<float>::FACE_TOP],
            (*this->renderData)[0].bbox[vislib::math::Cuboid<float>::FACE_FRONT]);
        out_bbox.SetObjectSpaceClipBox(out_bbox.ObjectSpaceBBox());
        out_bbox.MakeScaledWorld(1.0f);
    } else {
        out_bbox.SetObjectSpaceBBox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
        out_bbox.SetObjectSpaceClipBox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
    }
    return true;
}


bool FBOCompositor::Render(core::Call &call) {
    //this->num_render_nodes = this->numRenderNodesSlot.Param<core::param::IntParam>()->Value();
    this->resizeBuffers();

    core::view::CallRender3D *cr = dynamic_cast<core::view::CallRender3D*>(&call);
    if (cr == nullptr) return false;

    if (this->is_new_data.load()) {
        std::lock_guard<std::mutex> guard(this->swap_guard);

        // do upload
        for (int i = 0; i < this->num_render_nodes; i++) {
        //for (int i = this->num_render_nodes-1; i >= 0; i--) {
            auto &data = (*this->renderData)[i];

            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, this->color_textures[i]);
            glTexSubImage2D(GL_TEXTURE_2D, 0, data.viewport[0], data.viewport[1], data.viewport[2], data.viewport[3],
                GL_RGBA, GL_UNSIGNED_BYTE, data.color_buf.data());
            glBindTexture(GL_TEXTURE_2D, 0);

            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, this->depth_textures[i]);
            glTexSubImage2D(GL_TEXTURE_2D, 0, data.viewport[0], data.viewport[1], data.viewport[2], data.viewport[3],
                GL_DEPTH_COMPONENT, GL_FLOAT, data.depth_buf.data());
            glBindTexture(GL_TEXTURE_2D, 0);
        }

        //glBindTexture(GL_TEXTURE_2D, 0);

        this->is_new_data.store(false);
    }

    // do render
    
    glViewport(0, 0, this->fbo_width, this->fbo_height);
    glEnable(GL_DEPTH_TEST);

    // this is the apex of suck and must die
    GLfloat modelViewMatrix_column[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, modelViewMatrix_column);
    GLfloat projMatrix_column[16];
    glGetFloatv(GL_PROJECTION_MATRIX, projMatrix_column);
    GLfloat light_pos[4];
    glGetLightfv(GL_LIGHT0, GL_POSITION, light_pos);
    // end suck

    // enable shader
    glUseProgram(this->shader);

    glUniformMatrix4fv(glGetUniformLocation(this->shader, "modelview"), 1, GL_FALSE, modelViewMatrix_column);
    glUniformMatrix4fv(glGetUniformLocation(this->shader, "project"), 1, GL_FALSE, projMatrix_column);

    for (int i = 0; i < this->num_render_nodes; i++) {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, this->color_textures[i]);

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, this->depth_textures[i]);

        glUniform1i(glGetUniformLocation(this->shader, "color"), 0);
        glUniform1i(glGetUniformLocation(this->shader, "depth"), 1);

        glBindVertexArray(this->vao);
        glDrawArrays(GL_QUADS, 0, 4);
    }

    // disable shader
    glUseProgram(0);

    glDisable(GL_DEPTH_TEST);

    return true;
}


bool FBOCompositor::connectSocketCallback(core::param::ParamSlot &p) {
    if (this->is_connected) {
        for (int i = 0; i < this->ip_address.size(); i++) {
            this->zmq_socket.disconnect("tcp://" + this->ip_address[i]);
        }
        this->is_connected = false;
    }

    this->ip_address.clear();
    std::string addresses = std::string(this->ipAddressSlot.Param<core::param::StringParam>()->Value().PeekBuffer());
    // https://stackoverflow.com/questions/14265581/parse-split-a-string-in-c-using-string-delimiter-standard-c
    size_t pos = 0;
    std::string del(";");
    while ((pos = addresses.find(del)) != std::string::npos) {
        this->ip_address.push_back(addresses.substr(0, pos));
        addresses.erase(0, pos + del.length());
    }
    if (!addresses.empty()) {
        this->ip_address.push_back(addresses);
    }

    for (int i = 0; i < this->ip_address.size(); i++) {
        this->connectSocket(this->ip_address[i]);
    }

    //auto tmp = this->num_render_nodes;
    this->num_render_nodes = this->ip_address.size();
    //this->resizeBuffers(tmp);

    std::lock_guard<std::mutex> guard(this->swap_guard);

    this->renderData->resize(this->num_render_nodes);
    this->receiverData->resize(this->num_render_nodes);

    this->receiverThread = std::thread(&FBOCompositor::receiverCallback, this);

    return true;
}


void FBOCompositor::connectSocket(std::string &address) {
    try {
        this->zmq_socket.connect("tcp://" + address);
        this->is_connected = true;
    } catch (zmq::error_t e) {
        vislib::sys::Log::DefaultLog.WriteError("FBO Compositor: ZMQ error %s", e.what());
    }
}


void FBOCompositor::receiverCallback(void) {
    try {
        zmq::message_t msg;
        size_t idx = 0;
        std::vector<unsigned char> buffer;
        /*if (!this->zmq_socket.connected()) {
            this->ip_address = this->ipAddressSlot.Param<core::param::StringParam>()->Value();
            this->connectSocket(this->ip_address);
        }*/
        std::vector<bool> gotData(this->num_render_nodes, false);
        std::vector<std::queue<fbo_data>> fifo(this->num_render_nodes);
        while (true) {
            std::fill(gotData.begin(), gotData.end(), false);
            //for (int i = 0; i < this->num_render_nodes; i++) {
            //while (std::any_of(gotData.begin(), gotData.end(), [](const bool &a) {return !a; })) {
            while (std::any_of(fifo.begin(), fifo.end(), [](const std::queue<fbo_data> &a) {return (a.size()==0); })) {
                //auto &data = this->receiverData[i];

                vislib::sys::Log::DefaultLog.WriteInfo("FBOCompositor: Request frame\n");
                //this->zmq_socket.send("Frame", strlen("Frame"));
                this->zmq_socket.recv(&msg);
                fbo_data data;
                if (msg.size() < sizeof(data.viewport) + sizeof(int32_t) + sizeof(uint32_t)) {
                    throw std::runtime_error("FBOCompositor receiver thread: message (viewport) corrupted\n");
                }
                char *ptr = reinterpret_cast<char*>(msg.data());
                int32_t rank = *reinterpret_cast<int32_t*>(msg.data());
                ptr += sizeof(int32_t);
                data.fid = *reinterpret_cast<uint32_t*>(msg.data());
                ptr += sizeof(uint32_t);

                ASSERT(rank >= 0 && rank < receiverData->size());
                

                //auto &data = (*this->receiverData)[rank];
                
                memcpy(data.viewport, ptr, sizeof(data.viewport));
                //memcpy(this->viewport, data.viewport, sizeof(data.viewport));
                ptr += sizeof(data.viewport);
                int width = data.viewport[2] - data.viewport[0];
                int height = data.viewport[3] - data.viewport[1];
                memcpy(data.bbox, ptr, sizeof(data.bbox));
                ptr += sizeof(data.bbox);
                int datasize = (width)*(height)*4;
                if (width < 0 || height < 0 || datasize < 0 || msg.size() < 2*datasize + sizeof(data.viewport)) {
                    throw std::runtime_error("FBOCompositor receiver thread: message (data) corrupted\n");
                }
                data.color_buf.resize(datasize);
                data.depth_buf.resize(datasize);
                memcpy(data.color_buf.data(), ptr, datasize);
                ptr += datasize;
                memcpy(data.depth_buf.data(), ptr, datasize);

                fifo[rank].push(data);
                gotData[rank] = true;
            }

            /*uint32_t max_fid = 0;
            for (int i = 0; i < this->num_render_nodes; i++) {
                if (max_fid < fifo[i].front().fid) {
                    max_fid = fifo[i].front().fid;
                }
            }

            for (int i = 0; i < this->num_render_nodes; i++) {
                while (fifo[i].front().fid != max_fid && fifo[i].size() > 1) {
                    fifo[i].pop();
                }
                (*this->receiverData)[i] = fifo[i].front();
                fifo[i].pop();
            }*/

            for (int i = 0; i < this->num_render_nodes; i++) {
                (*this->receiverData)[i] = fifo[i].front();
                fifo[i].pop();
            }

            this->swapFBOData();
        }
    } catch (zmq::error_t e) {
        vislib::sys::Log::DefaultLog.WriteError("FBOCompositor: Unexpected failure in receiver thread: %s\n", e.what());
    }
}


void FBOCompositor::resizeBuffers(void) {
    //std::lock_guard<std::mutex> guard(this->swap_guard);

    //this->renderData.resize(this->num_render_nodes);
    //this->receiverData.resize(this->num_render_nodes);
    
    auto old_fbo_width = this->fbo_width;
    auto old_fbo_height = this->fbo_height;

    this->fbo_width = this->fboWidthSlot.Param<core::param::IntParam>()->Value();
    this->fbo_height = this->fboHeightSlot.Param<core::param::IntParam>()->Value();

    if (this->fbo_width != old_fbo_width || this->fbo_height != this->fbo_width/* || oldSize != this->num_render_nodes*/) {

        if (this->color_textures != nullptr) {
            glDeleteTextures(this->num_render_nodes, this->color_textures);
            delete[] this->color_textures;
        }
        if (this->depth_textures != nullptr) {
            glDeleteTextures(this->num_render_nodes, this->depth_textures);
            delete[] this->depth_textures;
        }

        this->color_textures = new GLuint[this->num_render_nodes];

        this->depth_textures = new GLuint[this->num_render_nodes];

        glGenTextures(this->num_render_nodes, this->color_textures);
        glGenTextures(this->num_render_nodes, this->depth_textures);

        for (int i = 0; i < this->num_render_nodes; i++) {
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, this->color_textures[i]);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, this->fbo_width, this->fbo_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glBindTexture(GL_TEXTURE_2D, 0);

            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, this->depth_textures[i]);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, this->fbo_width, this->fbo_height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glBindTexture(GL_TEXTURE_2D, 0);
        }
    }
}


void FBOCompositor::swapFBOData(void) {
    {
        std::lock_guard<std::mutex> guard(this->swap_guard);

        std::swap(this->renderData, this->receiverData);
        this->is_new_data.store(true);
    }

    //resizeBuffers(this->num_render_nodes);

    //updateNumRenderNodesCallback(this->ipAddressSlot);
}
