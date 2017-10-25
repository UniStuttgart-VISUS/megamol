using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;

namespace MegaMolConf {

    internal class GraphicalModuleMainViewDescriptor  : PropertyDescriptor {

        public GraphicalModuleMainViewDescriptor()
            : base("Is Main View", new Attribute[] { }) {
        }

        public override bool CanResetValue(object component) {
            return false;
        }

        public override Type ComponentType {
            get { return typeof(GraphicalModule); }
        }

        public override object GetValue(object component) {
            return Form1.isMainView((GraphicalModule)component);
        }

        public override bool IsReadOnly {
            get { return false; }
        }

        public override Type PropertyType {
            get { return typeof(bool); }
        }

        public override void ResetValue(object component) {
            // intentionally empty
        }

        public override void SetValue(object component, object value) {
            if ((bool)value) {
                Form1.setMainView((GraphicalModule)component);
            } else {
                Form1.removeMainView((GraphicalModule)component);
            }
        }

        public override bool ShouldSerializeValue(object component) {
            return true;
        }

        public override string DisplayName {
            get {
                return "Is Main View";
            }
        }

        public override string Description {
            get {
                return "Selects this View Module as Main View";
            }
        }

        public override string Category {
            get {
                return "General";
            }
        }

    }
}
