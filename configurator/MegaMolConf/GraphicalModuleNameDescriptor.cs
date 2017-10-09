using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;

namespace MegaMolConf {

    internal class GraphicalModuleNameDescriptor : PropertyDescriptor {

        public GraphicalModuleNameDescriptor()
            : base("Name", new Attribute[] { }) {
        }

        public override bool CanResetValue(object component) {
            return false;
        }

        public override Type ComponentType {
            get { return typeof(GraphicalModule); }
        }

        public override object GetValue(object component) {
            return ((GraphicalModule)component).Name;
        }

        public override bool IsReadOnly {
            get { return false; }
        }

        public override Type PropertyType {
            get { return typeof(string); }
        }

        public override void ResetValue(object component) {
            // intentionally empty
        }

        public override void SetValue(object component, object value) {
            ((GraphicalModule)component).Name = value.ToString();
        }

        public override bool ShouldSerializeValue(object component) {
            return true;
        }

        public override string DisplayName {
            get {
                return "Name";
            }
        }

        public override string Description {
            get {
                return "The instance name of the module";
            }
        }

        public override string Category {
            get {
                return "General";
            }
        }

    }

}
