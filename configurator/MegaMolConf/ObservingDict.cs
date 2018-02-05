using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace MegaMolConf {
    class ObservingDict<T> : IDictionary<TabPage, ObservableCollection<T>> {

        public delegate void ChangedItemsCallback(System.Collections.IList addedList, System.Collections.IList deletedList);

        public event ChangedItemsCallback OnChangedItems;

        public ObservingDict() {
            theDict = new Dictionary<TabPage, ObservableCollection<T>>();
            reverseDict = new Dictionary<ObservableCollection<T>, TabPage>();
        }

        public ICollection<TabPage> Keys => theDict.Keys;

        public ICollection<ObservableCollection<T>> Values => theDict.Values;

        public int Count => theDict.Count;

        public bool IsReadOnly => throw new NotImplementedException();

        public ObservableCollection<T> this[TabPage key] { get {
                return theDict[key];
            }
            set {
                theDict[key] = value;
                reverseDict[value] = key;
                value.CollectionChanged += Value_CollectionChanged;
            }
        }

        public bool ContainsKey(TabPage key) {
            return theDict.ContainsKey(key);
        }

        public void Add(TabPage key, ObservableCollection<T> value) {
            theDict.Add(key, value);
            reverseDict.Add(value, key);
            value.CollectionChanged += Value_CollectionChanged;
        }

        private void Value_CollectionChanged(object sender, System.Collections.Specialized.NotifyCollectionChangedEventArgs e) {
            Debug.WriteLine("got new items: " + e.NewItems);
            Debug.WriteLine("got removed items: " + e.OldItems);

            ChangedItemsCallback c = OnChangedItems;
            if ((e.NewItems != null || e.OldItems != null) && c != null) {
                c(e.NewItems, e.OldItems);
            }
            //Debug.WriteLine("for tabpage:" + reverseDict[sender])
        }

        public bool Remove(TabPage key) {
            // todo?
            reverseDict.Remove(theDict[key]);
            return theDict.Remove(key);
        }

        public bool TryGetValue(TabPage key, out ObservableCollection<T> value) {
            return theDict.TryGetValue(key, out value);
        }

        public void Add(KeyValuePair<TabPage, ObservableCollection<T>> item) {
            theDict.Add(item.Key, item.Value);
            reverseDict.Add(item.Value, item.Key);
            item.Value.CollectionChanged += Value_CollectionChanged;
        }

        public void Clear() {
            // todo ?
            theDict.Clear();
            reverseDict.Clear();
        }

        public bool Contains(KeyValuePair<TabPage, ObservableCollection<T>> item) {
            return theDict.Contains(item);
        }

        public void CopyTo(KeyValuePair<TabPage, ObservableCollection<T>>[] array, int arrayIndex) {
            if (array == null)
                throw new ArgumentNullException("array");
            ((ICollection<KeyValuePair<TabPage, ObservableCollection<T>>>)this).CopyTo(array, arrayIndex);
        }

        public bool Remove(KeyValuePair<TabPage, ObservableCollection<T>> item) {
            reverseDict.Remove(theDict[item.Key]);
            return theDict.Remove(item.Key);
        }

        public IEnumerator<KeyValuePair<TabPage, ObservableCollection<T>>> GetEnumerator() {
            return theDict.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator() {
            return theDict.GetEnumerator();
        }

        private Dictionary<TabPage, ObservableCollection<T>> theDict;
        private Dictionary<ObservableCollection<T>, TabPage> reverseDict;
    }
}
