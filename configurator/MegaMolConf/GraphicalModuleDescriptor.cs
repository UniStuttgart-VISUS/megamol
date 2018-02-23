using System;
using System.Collections.Generic;
using System.ComponentModel;

namespace MegaMolConf {

    internal class GraphicalModuleDescriptor : ICustomTypeDescriptor {

        private GraphicalModule gm;

        public GraphicalModuleDescriptor(GraphicalModule gm) {
            this.gm = gm;
        }

        public GraphicalModule Module {
            get { return gm; }
        }

        public AttributeCollection GetAttributes() {
            return TypeDescriptor.GetAttributes(this, true);
        }

        public string GetClassName() {
            return "GraphicalModuleDescriptor";
        }

        public string GetComponentName() {
            return "GraphicalModule";
        }

        public TypeConverter GetConverter() {
            return TypeDescriptor.GetConverter(this, true);
        }

        public EventDescriptor GetDefaultEvent() {
            return null;
        }

        public PropertyDescriptor GetDefaultProperty() {
            return new GraphicalModuleNameDescriptor();
        }

        public object GetEditor(Type editorBaseType) {
            return null;
        }

        public EventDescriptorCollection GetEvents(Attribute[] attributes) {
            return GetEvents();
        }

        public EventDescriptorCollection GetEvents() {
            return new EventDescriptorCollection(new EventDescriptor[] { });
        }

        public PropertyDescriptorCollection GetProperties(Attribute[] attributes) {
            return GetProperties();
        }

        public PropertyDescriptorCollection GetProperties() {
            List<PropertyDescriptor> pds = new List<PropertyDescriptor>();

            pds.Add(new GraphicalModuleNameDescriptor());
            if (gm.Module.IsViewModule) {
                pds.Add(new GraphicalModuleMainViewDescriptor());
            }

            foreach (KeyValuePair<Data.ParamSlot, string> p in gm.ParameterValues) {
                pds.Add(new GraphicalModuleParameterDescriptor(p.Key, gm.ParameterCmdLineness[p.Key]));
            }
            foreach (KeyValuePair<Data.ParamSlot, bool> p in gm.ParameterCmdLineness) {
                if (gm.ParameterValues.ContainsKey(p.Key)) continue;
                pds.Add(new GraphicalModuleButtonParameterDescriptor(p.Key, p.Value));
            }

            return new PropertyDescriptorCollection(pds.ToArray());
        }

        public object GetPropertyOwner(PropertyDescriptor pd) {
            return gm;
        }
    }

}
