#include "stdafx.h"
#include "PBSRenderer.h"

#include <fstream>

#include "mmcore/view/CallRender3D.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/utility/ResourceWrapper.h"
#include "mmcore/param/FloatParam.h"

#include "pbs/PBSDataCall.h"

using namespace megamol;
using namespace megamol::pbs;


PBSRenderer::PBSRenderer(void) : core::view::Renderer3DModule(),
getDataSlot("getData", "Data input slot"),
radiusParamSlot("radius", "Set radius of points") {
    this->getDataSlot.SetCompatibleCall<PBSDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

    this->radiusParamSlot << new core::param::FloatParam(0.5f, 0.000000000001f);
    this->MakeSlotAvailable(&this->radiusParamSlot);
}


PBSRenderer::~PBSRenderer(void) {
    this->Release();
}


bool PBSRenderer::printShaderInfoLog(GLuint shader) const {
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


bool PBSRenderer::printProgramInfoLog(GLuint shaderProg) const {
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


bool PBSRenderer::create(void) {
    glGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &this->max_ssbo_size);
    if (this->max_ssbo_size > 1048576000) {
        this->max_ssbo_size = 1048576000;
    }

    // create shader storage buffers
    glGenBuffers(1, &this->x_buffer);
    glGenBuffers(1, &this->y_buffer);
    glGenBuffers(1, &this->z_buffer);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->x_buffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, this->max_ssbo_size, nullptr, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, this->x_buffer_base, this->x_buffer);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->y_buffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, this->max_ssbo_size, nullptr, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, this->y_buffer_base, this->y_buffer);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->z_buffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, this->max_ssbo_size, nullptr, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, this->z_buffer_base, this->z_buffer);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    // create shader
    auto vert_shader = glCreateShader(GL_VERTEX_SHADER);
    auto frag_shader = glCreateShader(GL_FRAGMENT_SHADER);

    auto path_to_vert = core::utility::ResourceWrapper::getFileName(this->GetCoreInstance()->Configuration(), "pbssphere.vert.glsl");
    auto path_to_frag = core::utility::ResourceWrapper::getFileName(this->GetCoreInstance()->Configuration(), "pbssphere.frag.glsl");

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


void PBSRenderer::release(void) {
    glDeleteBuffers(1, &this->x_buffer);
    glDeleteBuffers(1, &this->y_buffer);
    glDeleteBuffers(1, &this->z_buffer);

    glDeleteProgram(this->shader);
}


bool PBSRenderer::Render(core::Call &call) {
    core::view::CallRender3D *cr = dynamic_cast<core::view::CallRender3D*>(&call);
    if (cr == nullptr) return false;

    PBSDataCall *c2 = this->getDataSlot.CallAs<PBSDataCall>();
    if (c2 == nullptr) return false;

    auto data = c2->GetData().lock();

    auto x_data = data->GetX().lock();
    auto y_data = data->GetY().lock();
    auto z_data = data->GetZ().lock();

    if (x_data->size() != y_data->size() || x_data->size() != z_data->size() || y_data->size() != z_data->size()) {
        return false;
    }

    int64_t data_chunk_size = this->max_ssbo_size / sizeof((*x_data)[0]);
    int num_chunks = 1;
    if (data_chunk_size > x_data->size()) {
        data_chunk_size = x_data->size();
    } else {
        num_chunks = std::ceil(x_data->size() / data_chunk_size);
    }

    auto radius = this->radiusParamSlot.Param<core::param::FloatParam>()->Value();

    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

    float viewportStuff[4];
    glGetFloatv(GL_VIEWPORT, viewportStuff);
    glPointSize(vislib::math::Max(viewportStuff[2], viewportStuff[3]));
    if (viewportStuff[2] < 1.0f) viewportStuff[2] = 1.0f;
    if (viewportStuff[3] < 1.0f) viewportStuff[3] = 1.0f;
    viewportStuff[2] = 2.0f / viewportStuff[2];
    viewportStuff[3] = 2.0f / viewportStuff[3];

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

    glUniform4fv(glGetUniformLocation(this->shader, "viewAttr"), 1, viewportStuff);
    glUniform3fv(glGetUniformLocation(this->shader, "camIn"), 1, cr->GetCameraParameters()->Front().PeekComponents());
    glUniform3fv(glGetUniformLocation(this->shader, "camRight"), 1, cr->GetCameraParameters()->Right().PeekComponents());
    glUniform3fv(glGetUniformLocation(this->shader, "camUp"), 1, cr->GetCameraParameters()->Up().PeekComponents());
    glUniform1f(glGetUniformLocation(this->shader, "scaling"), 1.0f);
    glUniform4f(glGetUniformLocation(this->shader, "inConst"), radius, 0.0f, 0.0f, 0.0f);

    glUniformMatrix4fv(glGetUniformLocation(this->shader, "modelview"), 1, GL_FALSE, modelViewMatrix_column);
    glUniformMatrix4fv(glGetUniformLocation(this->shader, "project"), 1, GL_FALSE, projMatrix_column);
    glUniform4fv(glGetUniformLocation(this->shader, "lightPos_u"), 1, light_pos);

    for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
        int64_t current_chunk_size = data_chunk_size;
        if (chunk_idx == num_chunks - 1) {
            current_chunk_size = x_data->size() - (chunk_idx*data_chunk_size);
        }

        // upload data
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->x_buffer);
        auto ptr = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_WRITE_ONLY);
        auto data_ptr = &((x_data->data())[chunk_idx*current_chunk_size]);
        memcpy(ptr, data_ptr, sizeof((*x_data)[0])*current_chunk_size);
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->y_buffer);
        ptr = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_WRITE_ONLY);
        data_ptr = &((y_data->data())[chunk_idx*current_chunk_size]);
        memcpy(ptr, data_ptr, sizeof((*y_data)[0])*current_chunk_size);
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, this->z_buffer);
        ptr = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_WRITE_ONLY);
        data_ptr = &((z_data->data())[chunk_idx*current_chunk_size]);
        memcpy(ptr, data_ptr, sizeof((*z_data)[0])*current_chunk_size);
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

        // render data
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, this->x_buffer_base, this->x_buffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, this->y_buffer_base, this->y_buffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, this->z_buffer_base, this->z_buffer);
        glDrawArrays(GL_POINTS, 0, current_chunk_size);
    }

    glUseProgram(0);

    glEnable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);

    return true;
}


bool PBSRenderer::GetExtents(core::Call &call) {
    core::view::CallRender3D *cr = dynamic_cast<core::view::CallRender3D*>(&call);
    if (cr == nullptr) return false;

    PBSDataCall *c2 = this->getDataSlot.CallAs<PBSDataCall>();

    if ((c2 != nullptr) && ((*c2)(0))) {
        cr->SetTimeFramesCount(1);

        auto in_bbox = c2->GetGlobalBBox().lock();

        auto &out_bbox = cr->AccessBoundingBoxes();

        out_bbox.SetObjectSpaceBBox(in_bbox.get()[0], in_bbox.get()[1], in_bbox.get()[2],
            in_bbox.get()[3], in_bbox.get()[4], in_bbox.get()[5]);
        out_bbox.SetObjectSpaceClipBox(in_bbox.get()[0], in_bbox.get()[1], in_bbox.get()[2],
            in_bbox.get()[3], in_bbox.get()[4], in_bbox.get()[5]);
        out_bbox.MakeScaledWorld(1.0f);
    } else {
        cr->SetTimeFramesCount(1);
        cr->AccessBoundingBoxes().Clear();
    }

    return true;
}
