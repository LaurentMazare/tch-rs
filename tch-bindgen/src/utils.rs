use crate::common::*;

pub trait StrExt {
    fn is_prefix_of(&self, haystack: &str) -> bool;
    fn is_suffix_of(&self, haystack: &str) -> bool;
}

impl StrExt for &str {
    fn is_prefix_of(&self, haystack: &str) -> bool {
        let len = self.len();
        haystack.get(0..len).map_or(false, |prefix| *self == prefix)
    }

    fn is_suffix_of(&self, haystack: &str) -> bool {
        let len = self.len();
        let end = haystack.len();
        let begin = match end.checked_sub(len) {
            Some(begin) => begin,
            None => return false,
        };
        haystack
            .get(begin..end)
            .map_or(false, |suffix| *self == suffix)
    }
}

impl StrExt for String {
    fn is_prefix_of(&self, haystack: &str) -> bool {
        self.as_str().is_prefix_of(haystack)
    }

    fn is_suffix_of(&self, haystack: &str) -> bool {
        self.as_str().is_suffix_of(haystack)
    }
}

pub trait IteratorExt: Iterator {
    fn into_group_index_map<K, V>(self) -> IndexMap<K, Vec<V>>
    where
        Self: Iterator<Item = (K, V)>,
        K: Hash + Eq;
}

impl<I: Iterator> IteratorExt for I {
    fn into_group_index_map<K, V>(self) -> IndexMap<K, Vec<V>>
    where
        Self: Iterator<Item = (K, V)>,
        K: Hash + Eq,
    {
        let mut map = IndexMap::<K, Vec<V>>::new();
        self.for_each(|(key, value)| {
            let values = map.entry(key).or_insert_with(|| vec![]);
            values.push(value);
        });
        map
    }
}
