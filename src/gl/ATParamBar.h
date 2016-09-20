/*
 * gl/ATParamBar.h
 *
 * Copyright (C) 2016 MegaMol Team
 * Alle Rechte vorbehalten.
 */
#ifndef MEGAMOLCON_GL_ATPARAMBAR_H_INCLUDED
#define MEGAMOLCON_GL_ATPARAMBAR_H_INCLUDED
#pragma once

#ifdef HAS_ANTTWEAKBAR
#include "gl/ATBar.h"
#include <vector>
#include <memory>
#include <sstream>
#include <string>
#include "mmcore/api/MegaMolCore.h"
#include "CoreHandle.h"
#include "vislib/String.h"

namespace megamol {
namespace console {
namespace gl {

    /** Base class for AntTweakBar bars */
    class ATParamBar : public ATBar {
    public:
        ATParamBar(void *hCore);
        virtual ~ATParamBar();
        void Update();
    private:

        class Param {
        public:
            Param(const std::string& name);
            virtual ~Param();

            inline CoreHandle& Handle() {
                return hParam;
            }
            inline CoreHandle const& Handle() const {
                return hParam;
            }
            inline std::string const& GetName() const {
                return name;
            }
            inline void SetID(unsigned int idval) {
                id = idval;
            }
            inline unsigned int GetID(void) const {
                return id;
            }
            inline std::string GetKey() const {
                return std::string("E") + std::to_string(id);
            }

        protected:
            Param(Param&& src);

        private:
            Param(const Param& src) = delete;
            Param& operator=(const Param& src) = delete;

            unsigned int id;
            std::string name;
            CoreHandle hParam;
        };

        class ButtonParam : public Param {
        public:
            ButtonParam(std::shared_ptr<Param> src, TwBar* bar, std::stringstream& def, const std::vector<unsigned char>& desc);
        private:
            static void TW_CALL twCallback(ButtonParam *prm);
        };

        class ValueParam : public Param {
        protected:
            ValueParam(std::shared_ptr<Param> src, TwBar* bar, TwType type, std::stringstream& def);
            virtual void Set(const void *value) = 0;
            virtual void Get(void *value) = 0;
        private:
            static void TW_CALL twSetCallback(const void *value, ValueParam *clientData);
            static void TW_CALL twGetCallback(void *value, ValueParam *clientData);
        };

        class StringParam : public ValueParam {
        public:
            StringParam(std::shared_ptr<Param> src, TwBar* bar, std::stringstream& def, const std::vector<unsigned char>& desc);
            virtual void Set(const void *value);
            virtual void Get(void *value);
        };

        class BoolParam : public ValueParam {
        public:
            BoolParam(std::shared_ptr<Param> src, TwBar* bar, std::stringstream& def, const std::vector<unsigned char>& desc);
            virtual void Set(const void *value);
            virtual void Get(void *value);
        };

        class EnumParam : public ValueParam {
        public:
            EnumParam(std::shared_ptr<Param> src, TwBar* bar, std::stringstream& def, const std::vector<unsigned char>& desc);
            virtual void Set(const void *value);
            virtual void Get(void *value);
        private:
            static TwType makeMyEnumType(void* hParam, const std::vector<unsigned char>& desc);
            static void parseEnumDesc(std::vector<TwEnumVal>& outValues, const std::vector<unsigned char>& desc);
            static std::vector<std::shared_ptr<vislib::StringA> > enumStrings;
            std::vector<TwEnumVal> values;
        };

        class FloatParam : public ValueParam {
        public:
            FloatParam(std::shared_ptr<Param> src, TwBar* bar, std::stringstream& def, const std::vector<unsigned char>& desc);
            virtual void Set(const void *value);
            virtual void Get(void *value);
        };

        class IntParam : public ValueParam {
        public:
            IntParam(std::shared_ptr<Param> src, TwBar* bar, std::stringstream& def, const std::vector<unsigned char>& desc);
            virtual void Set(const void *value);
            virtual void Get(void *value);
        };

        class Vec3fParam : public ValueParam {
        public:
            Vec3fParam(std::shared_ptr<Param> src, TwBar* bar, std::stringstream& def, const std::vector<unsigned char>& desc);
            virtual void Set(const void *value);
            virtual void Get(void *value);
        private:
            static TwType makeMyStructType();
            static void TW_CALL mySummaryCallback(char *summaryString, size_t summaryMaxLength, const void *value, void *summaryClientData);
        };

        struct group {
            unsigned int id;
            std::string name;
            std::vector<std::shared_ptr<group> > subgroups;
            std::vector<std::shared_ptr<Param> > params;

            group* findSubGroup(std::string name) const;
            Param* findParam(std::string name) const;
            bool structEqual(group* rhs) const;
            void sort();
            void setIds(unsigned int &i);
        };

        static void MEGAMOLCORE_CALLBACK mmcCollectParams(const char *str, gl::ATParamBar::group *data);

        void addParamVariables(std::string name, group *grp);
        void addParamVariable(std::shared_ptr<Param>& param, std::string name_prefix, std::string name, unsigned int grpId);

        void *hCore;
        group root;
    };

} /* end namespace gl */
} /* end namespace console */
} /* end namespace megamol */

#endif /* HAS_ANTTWEAKBAR */
#endif /* MEGAMOLCON_GL_ATPARAMBAR_H_INCLUDED */
