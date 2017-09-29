using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;

namespace MegaMolConf {

    internal class GraphicalModuleButtonParameterDescriptor : GraphicalModuleParameterDescriptor {

        public GraphicalModuleButtonParameterDescriptor(Data.ParamSlot p, bool useInCmdLine)
            : base(p, useInCmdLine) {
        }

        public override bool CanResetValue(object component) {
            return false;
        }

        public override object GetValue(object component) {
            return string.Empty;
        }

        public override bool IsReadOnly {
            get { return true; }
        }

        public override Type PropertyType {
            get { return typeof(string); }
        }

        public override void ResetValue(object component) {
            // intentionally empty
        }

        public override void SetValue(object component, object value) {
            // intentionally empty
        }

        public override bool ShouldSerializeValue(object component) {
            return false;
        }

        public override string Category {
            get {
                return "Button Parameters";
            }
        }

    }

}
