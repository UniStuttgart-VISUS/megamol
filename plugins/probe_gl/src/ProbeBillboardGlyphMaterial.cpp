#include "ProbeBillboardGlyphMaterial.h"

#include "mesh/MeshCalls.h"
#include "ProbeCalls.h"

megamol::probe_gl::ProbeBillboardGlyphMaterial::ProbeBillboardGlyphMaterial() 
    : m_glyph_images_slot("GetProbes", "Slot for accessing a probe collection"), m_glyph_images_slot_cached_hash(0) {

    this->m_glyph_images_slot.SetCompatibleCall<mesh::CallImageDescription>();
    this->MakeSlotAvailable(&this->m_glyph_images_slot);

}

megamol::probe_gl::ProbeBillboardGlyphMaterial::~ProbeBillboardGlyphMaterial() {}

bool megamol::probe_gl::ProbeBillboardGlyphMaterial::create() {
    // create shader program
    vislib::graphics::gl::ShaderSource vert_shader_src;
    vislib::graphics::gl::ShaderSource frag_shader_src;

    vislib::StringA shader_base_name("ProbeGlyph");
    vislib::StringA vertShaderName = shader_base_name + "::vertex";
    vislib::StringA fragShaderName = shader_base_name + "::fragment";

    this->instance()->ShaderSourceFactory().MakeShaderSource(vertShaderName.PeekBuffer(), vert_shader_src);
    this->instance()->ShaderSourceFactory().MakeShaderSource(fragShaderName.PeekBuffer(), frag_shader_src);

    try {
        m_billboard_glyph_prgm = std::make_unique<GLSLShader>();
        m_billboard_glyph_prgm->Create(
            vert_shader_src.Code(), vert_shader_src.Count(), frag_shader_src.Code(), frag_shader_src.Count());
    } catch (vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "Unable to compile %s (@%s):\n%s\n",
            shader_base_name.PeekBuffer(),
            vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(ce.FailedAction()),
            ce.GetMsgA());
        // return false;
    } catch (vislib::Exception e) {
        vislib::sys::Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_ERROR, "Unable to compile %s:\n%s\n", shader_base_name.PeekBuffer(), e.GetMsgA());
        // return false;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_ERROR, "Unable to compile %s: Unknown exception\n", shader_base_name.PeekBuffer());
        // return false;
    }

    // m_drawToScreen_prgm

    return true;
}

bool megamol::probe_gl::ProbeBillboardGlyphMaterial::getDataCallback(core::Call& caller) {

    mesh::CallGPUMaterialData* lhs_mtl_call = dynamic_cast<mesh::CallGPUMaterialData*>(&caller);
    if (lhs_mtl_call == NULL) return false;

    // no incoming material -> use your own material storage
    if (lhs_mtl_call->getData() == nullptr) lhs_mtl_call->setData(this->m_gpu_materials);
    std::shared_ptr<mesh::GPUMaterialCollecton> mtl_collection = lhs_mtl_call->getData();

    // if there is a material connection to the right, pass on the material collection
    mesh::CallGPUMaterialData* rhs_mtl_call = this->m_mtl_callerSlot.CallAs<mesh::CallGPUMaterialData>();
    if (rhs_mtl_call != NULL) rhs_mtl_call->setData(mtl_collection);

    
    mesh::CallImage* ic = this->m_glyph_images_slot.CallAs<mesh::CallImage>();
    if (ic == NULL) return false;

    auto image_meta_data = ic->getMetaData();

    if (image_meta_data.m_data_hash > this->m_glyph_images_slot_cached_hash)
    {
        this->m_glyph_images_slot_cached_hash = image_meta_data.m_data_hash;

        if (!(*ic)(0)) return false;
        auto img_data = ic->getData();

        auto images = img_data->accessImages();

        for (auto img : images)
        {
            glowl::TextureLayout tex_layout;
            tex_layout.width = img.width;
            tex_layout.height = img.height;
            tex_layout.depth = 1;
            tex_layout.levels = 1;
            tex_layout.internal_format = mesh::ImageDataAccessCollection::convertToGLInternalFormat(img.format);

            auto new_tex_ptr = std::make_unique<glowl::Texture2D>("ProbeGlyph",tex_layout,img.data);


        }

    }


    return true; 
}

bool megamol::probe_gl::ProbeBillboardGlyphMaterial::getMetaDataCallback(core::Call& caller) {

    //if (!mesh::AbstractGPUMaterialDataSource::getMetaDataCallback(caller)) return false;

    auto glyph_image_call = m_glyph_images_slot.CallAs<mesh::CallImage>();
    if (!(*glyph_image_call)(1)) return false;

    return true;
}
