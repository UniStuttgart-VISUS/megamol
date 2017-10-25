using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing.Design;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.Windows.Forms.Design;

namespace MegaMolConf {

    internal class EnumParamEditor : UITypeEditor {

        internal class ElementTypeConverter : TypeConverter {
            public override bool CanConvertTo(ITypeDescriptorContext context, Type destinationType) {
                return typeof(string) == destinationType;
            }
            public override object ConvertTo(ITypeDescriptorContext context, CultureInfo culture, object value, Type destinationType) {
                if (typeof(string) == destinationType) {
                    return ((Element)value).DisplayName;
                }
                return string.Empty;
            }
        }

        [TypeConverter(typeof(ElementTypeConverter))]
        internal class Element {
            public int Value { get; set; }
            public string Name { get; set; }
            public Element(int v, string n) {
                this.Value = v;
                this.Name = n;
            }
            public string DisplayName {
                get { return string.Format("{0}: {1}", this.Value, this.Name); }
            }
            public override string ToString() {
                return this.Value.ToString();
            }
        };

        private Data.ParamType.Enum e;

        private IWindowsFormsEditorService _editorService;

        public EnumParamEditor(Data.ParamType.Enum e) {
            this.e = e;
        }

        public override UITypeEditorEditStyle GetEditStyle(ITypeDescriptorContext context) {
            // drop down mode (we'll host a listbox in the drop down)
            return UITypeEditorEditStyle.DropDown;
        }

        public override object EditValue(ITypeDescriptorContext context, IServiceProvider provider, object value) {
            _editorService = (IWindowsFormsEditorService)provider.GetService(typeof(IWindowsFormsEditorService));

            // use a list box
            ListBox lb = new ListBox();
            lb.SelectionMode = SelectionMode.One;
            lb.SelectedValueChanged += OnListBoxSelectedValueChanged;

            lb.DisplayMember = "DisplayName";

            for (int i = 0; i < this.e.Values.Length; i++) {
                Element e = new Element(this.e.Values[i], this.e.ValueNames[i]);
                int index = lb.Items.Add(e);
                if (e.DisplayName == ((Element)value).DisplayName) {
                    lb.SelectedIndex = index;
                }
            }

            // show this model stuff
            _editorService.DropDownControl(lb);
            if (lb.SelectedItem == null) // no selection, return the passed-in value as is
                return value;

            return lb.SelectedItem;
        }

        private void OnListBoxSelectedValueChanged(object sender, EventArgs e) {
            // close the drop down as soon as something is clicked
            _editorService.CloseDropDown();
        }

    }

}
