/*
 * ProjectParser.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "mmcore/utility/ProjectParser.h"
#include "mmcore/factories/CallDescriptionManager.h"
#include "mmcore/factories/ModuleDescriptionManager.h"
#include "vislib/VersionNumber.h"
#include "vislib/assert.h"
#include "vislib/memutils.h"


using namespace megamol::core;
using namespace megamol::core::utility::xml;


/*
 * utility::ProjectParser::ProjectParser
 */
utility::ProjectParser::ProjectParser(CoreInstance* coreInst)
        : xml::ConditionalParser()
        , core(coreInst)
        , vd()
        , jd()
        , viewDescs()
        , jobDescs() {
    // intentionally empty
}


/*
 * utility::ProjectParser::~ProjectParser
 */
utility::ProjectParser::~ProjectParser(void) {
    this->core = nullptr; // do not delete
    this->vd.reset();
    this->jd.reset();
    this->viewDescs.Clear();
    this->jobDescs.Clear();
}


/*
 * utility::ProjectParser::CheckBaseTag
 */
bool utility::ProjectParser::CheckBaseTag(const utility::xml::XmlReader& reader) {
    if (!reader.BaseTag().Equals(MMXML_STRING("MegaMol"))) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(1, "Project file does not specify <MegaMol> as base tag");
        return false;
    }

    bool typeValid = false;
    bool versionValid = false;

    const vislib::Array<XmlReader::XmlAttribute>& attrib = reader.BaseTagAttributes();
    for (unsigned int i = 0; i < attrib.Count(); i++) {
        const XmlReader::XmlAttribute& attr = attrib[i];

        if (attr.Key().Equals(MMXML_STRING("type"))) {
            if (attr.Value().Equals(MMXML_STRING("project"))) {
                typeValid = true;
            }

        } else if (attr.Key().Equals(MMXML_STRING("version"))) {
            vislib::VersionNumber ver;
            if (ver.Parse(attr.Value()) >= 1) {
                if (ver < vislib::VersionNumber(1, 0)) {
                    versionValid = false; // pre 1.0 does not exist!
                } else if (ver < vislib::VersionNumber(1, 1)) {
                    versionValid = true; // 1.0.x.x
                    this->setConditionalParserVersion(0);
                } else if (ver < vislib::VersionNumber(1, 2)) {
                    versionValid = true; // 1.1.x.x
                    this->setConditionalParserVersion(1);
                } else {
                    versionValid = false; // >= 1.2 does not exist yet!
                }
            }
        } else {
            this->Warning(vislib::StringA("Ignoring unexpected base tag attribute ") + vislib::StringA(attr.Key()));
        }
    }

    if (!typeValid) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(1, "base tag attribute \"type\" not present or invalid.");
        return false;
    }
    if (!versionValid) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(
            1, "base tag attribute \"version\" not present or invalid.");
        return false;
    }

    return typeValid && versionValid;
}


/*
 * utility::ProjectParser::StartTag
 */
bool utility::ProjectParser::StartTag(unsigned int num, unsigned int level, const XML_Char* name,
    const XML_Char** attrib, utility::xml::XmlReader::ParserState state,
    utility::xml::XmlReader::ParserState& outChildState, utility::xml::XmlReader::ParserState& outEndTagState,
    utility::xml::XmlReader::ParserState& outPostEndTagState) {

    if (ConditionalParser::StartTag(
            num, level, name, attrib, state, outChildState, outEndTagState, outPostEndTagState)) {
        return true; // handled by base class
    }

    if (MMXML_STRING("view").Equals(name)) {
        if (this->vd != NULL) {
            this->Error("\"view\" tag nested within another \"view\" tag ignored");
            return true;
        }
        if (this->jd != NULL) {
            this->Error("\"view\" tag nested within \"job\" tag ignored");
            return true;
        }
        const XML_Char* vdname = NULL;
        const XML_Char* viewmodname = NULL;
        for (int i = 0; attrib[i]; i += 2) {
            if (MMXML_STRING("name").Equals(attrib[i])) {
                vdname = attrib[i + 1];
            } else if (MMXML_STRING("viewmod").Equals(attrib[i])) {
                viewmodname = attrib[i + 1];
            } else {
                this->WarnUnexpectedAttribut(name, attrib[i]);
            }
        }
        if (vdname == NULL) {
            this->Error("\"view\" tag without a name ignored");
            return true;
        }

        this->vd = std::make_shared<ViewDescription>(vislib::StringA(vdname));

        if (viewmodname == NULL) {
            this->Warning("\"view\" tag without \"viewmod\" might result in unexpected behaviour");
        } else {
            vd->SetViewModuleID(vislib::StringA(viewmodname));
        }

        return true;
    }

    if (MMXML_STRING("job").Equals(name)) {
        if (this->jd != NULL) {
            this->Error("\"job\" tag nested within another \"job\" tag ignored");
            return true;
        }
        if (this->vd != NULL) {
            this->Error("\"job\" tag nested within \"view\" tag ignored");
            return true;
        }
        const XML_Char* jdname = NULL;
        const XML_Char* jobmodname = NULL;
        for (int i = 0; attrib[i]; i += 2) {
            if (MMXML_STRING("name").Equals(attrib[i])) {
                jdname = attrib[i + 1];
            } else if (MMXML_STRING("jobmod").Equals(attrib[i])) {
                jobmodname = attrib[i + 1];
            } else {
                this->WarnUnexpectedAttribut(name, attrib[i]);
            }
        }
        if (jdname == NULL) {
            this->Error("\"job\" tag without a name ignored");
            return true;
        }

        this->jd = std::make_shared<JobDescription>(vislib::StringA(jdname));

        if (jobmodname == NULL) {
            this->Warning("\"job\" tag without \"jobmod\" might result in unexpected behaviour");
        } else {
            jd->SetJobModuleID(vislib::StringA(jobmodname));
        }

        return true;
    }

    if (MMXML_STRING("module").Equals(name)) {
        if ((this->vd == NULL) && (this->jd == NULL)) {
            this->Error("\"module\" tag outside a \"view\" tag or a \"job\" tag ignored");
            return true;
        }
        const XML_Char* className = NULL;
        const XML_Char* instName = NULL;
        for (int i = 0; attrib[i]; i += 2) {
            if (MMXML_STRING("class").Equals(attrib[i])) {
                className = attrib[i + 1];
            } else if (MMXML_STRING("name").Equals(attrib[i])) {
                instName = attrib[i + 1];
            } else {
                this->WarnUnexpectedAttribut(name, attrib[i]);
            }
        }
        if (instName == NULL) {
            this->Error("\"module\" tag without a name ignored");
            return true;
        }
        if (className == NULL) {
            this->Error("\"module\" tag without a class ignored");
            return true;
        }
        factories::ModuleDescription::ptr md =
            this->core->GetModuleDescriptionManager().Find(vislib::StringA(className));
        if (md == NULL) {
            vislib::StringA msg;
            msg.Format("\"module\" class \"%s\" not found", vislib::StringA(className).PeekBuffer());
            this->Error(msg.PeekBuffer());
            return true;
        }
        this->modName = instName;
        if (this->vd != NULL) {
            this->vd->AddModule(md, this->modName);
        } else {
            ASSERT(this->jd != NULL);
            this->jd->AddModule(md, this->modName);
        }
        return true;
    }

    if (MMXML_STRING("call").Equals(name)) {
        if ((this->vd == NULL) && (this->jd == NULL)) {
            this->Error("\"call\" tag outside a \"view\" tag or a \"job\" tag ignored");
            return true;
        }
        const XML_Char* className = NULL;
        const XML_Char* fromName = NULL;
        const XML_Char* toName = NULL;
        const XML_Char* profilingSetting = NULL;
        for (int i = 0; attrib[i]; i += 2) {
            if (MMXML_STRING("class").Equals(attrib[i])) {
                className = attrib[i + 1];
            } else if (MMXML_STRING("from").Equals(attrib[i])) {
                fromName = attrib[i + 1];
            } else if (MMXML_STRING("to").Equals(attrib[i])) {
                toName = attrib[i + 1];
            } else if (MMXML_STRING("profile").Equals(attrib[i])) {
                profilingSetting = attrib[i + 1];
            } else {
                this->WarnUnexpectedAttribut(name, attrib[i]);
            }
        }
        if (className == NULL) {
            this->Error("\"call\" tag without a class ignored");
            return true;
        }
        if (fromName == NULL) {
            this->Error("\"call\" tag without a from attribute ignored");
            return true;
        }
        if (toName == NULL) {
            this->Error("\"call\" tag without a to attribute ignored");
            return true;
        }
        factories::CallDescription::ptr cd = this->core->GetCallDescriptionManager().Find(vislib::StringA(className));
        if (cd == NULL) {
            this->Error("\"call\" class not found");
            return true;
        }
        bool doProfiling = false;
        try {
            doProfiling = vislib::CharTraits<XML_Char>::ParseBool(profilingSetting);
        } catch (...) {}
        if (this->vd != NULL) {
            this->vd->AddCall(cd, vislib::StringA(fromName), vislib::StringA(toName), doProfiling);
        } else {
            ASSERT(this->jd != NULL);
            this->jd->AddCall(cd, vislib::StringA(fromName), vislib::StringA(toName), doProfiling);
        }
        return true;
    }

    if (MMXML_STRING("param").Equals(name)) {
        if ((this->vd == NULL) && (this->jd == NULL)) {
            this->Error("\"param\" tag outside a \"view\" tag a \"job\" tag ignored");
            return true;
        }
        const XML_Char* name = NULL;
        const XML_Char* value = NULL;
        for (int i = 0; attrib[i]; i += 2) {
            if (MMXML_STRING("name").Equals(attrib[i])) {
                name = attrib[i + 1];
            } else if (MMXML_STRING("value").Equals(attrib[i])) {
                value = attrib[i + 1];
            } else {
                this->WarnUnexpectedAttribut(name, attrib[i]);
            }
        }
        if (name == NULL) {
            this->Error("\"param\" tag without a name ignored");
            return true;
        }
        if (value == NULL) {
            this->Error("\"param\" tag without a value ignored");
            return true;
        }
        if (this->modName.IsEmpty()) {
            if (this->vd != NULL) {
                this->vd->AddParamValue(vislib::StringA(name), vislib::TString(value));
            } else {
                ASSERT(this->jd != NULL);
                this->jd->AddParamValue(vislib::StringA(name), vislib::TString(value));
            }
        } else {
            vislib::StringA n(name);
            if (!n.StartsWith("::")) {
                n.Prepend("::");
                n.Prepend(this->modName);
            }
            if (this->vd != NULL) {
                this->vd->AddParamValue(n, vislib::TString(value));
            } else {
                ASSERT(this->jd != NULL);
                this->jd->AddParamValue(n, vislib::TString(value));
            }
        }
        return true;
    }

    // TODO: Implement instance tag
    //       etc.

    return false;
}


/*
 * utility::ProjectParser::EndTag
 */
bool utility::ProjectParser::EndTag(unsigned int num, unsigned int level, const XML_Char* name,
    utility::xml::XmlReader::ParserState state, utility::xml::XmlReader::ParserState& outPostEndTagState) {
    if (ConditionalParser::EndTag(num, level, name, state, outPostEndTagState)) {
        return true; // handled by base class
    }

    if (MMXML_STRING("view").Equals(name)) {
        ASSERT(this->vd != NULL);
        this->viewDescs.Add(this->vd);
        this->vd = NULL;
        return true;
    }
    if (MMXML_STRING("job").Equals(name)) {
        ASSERT(this->jd != NULL);
        this->jobDescs.Add(this->jd);
        this->jd = NULL;
        return true;
    }
    if (MMXML_STRING("module").Equals(name)) {
        this->modName.Clear();
        return true;
    }
    if (MMXML_STRING("call").Equals(name) || MMXML_STRING("param").Equals(name)
        /*|| MMXML_STRING("instance").Equals(name)*/) {
        return true;
    }

    return false; // unhandled.
}
