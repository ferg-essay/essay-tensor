use core::fmt;
use std::{collections::HashMap, str::{Chars, FromStr}, iter::Peekable, ops::Deref, sync::Arc};

pub(crate) fn read_config() -> Config
{
    let data = include_bytes!("essay-plot.rc");

    let data = String::from_utf8(data.to_vec()).unwrap();
    // TODO: Cursor with line info
    let mut iter: Peekable<Chars> = data.chars().peekable();

    let mut config = Config::new();

    while read_line(&mut config, &mut iter) {
    }

    config
}

fn read_line(config: &mut Config, iter: &mut Peekable<Chars>) -> bool {
    let ch = match skip_whitespace(iter) {
        Some(ch) => ch,
        None => { return false; }
    };

    let name = match read_identifier(ch, iter) {
        Some(name) => name,
        None => { return true; }
    };

    match iter.next() {
        Some(':') => {},
        _ => panic!("expected ':'")
    }

    let value = read_value(iter);

    config.add_value(name, value);

    true
}

fn read_identifier(ch: char, iter: &mut Peekable<Chars>) -> Option<String> {
    if ch == '#' {
        skip_comment(iter);
        None
    } else if ch.is_alphabetic() {
        let mut name = String::new();
        name.push(ch);

        while let Some(ch) = iter.peek() {
            if ch.is_whitespace() || *ch == '#' || *ch == ':' {
                return Some(name);
            }

            name.push(*ch);
            iter.next();
        }

        Some(name)
    } else {
        panic!("Unexpected character {:?}", ch);
    }
}

fn read_value(iter: &mut Peekable<Chars>) -> String {
    let mut value = String::new();

    let ch = match skip_space(iter) {
        Some(ch) => ch,
        None => return value,
    };

    if ch == '"' {
        return read_string_value(iter);
    }

    value.push(ch);

    while let Some(ch) = iter.next() {
        match ch {
            '\r' | '\n' => { return value }
            '#' => { skip_comment(iter); return value }
            _ => { value.push(ch); }
        }
    }

    value
}

fn read_string_value(iter: &mut Peekable<Chars>) -> String {
    let mut value = String::new();

    while let Some(ch) = iter.next() {
        match ch {
            '\r' | '\n' => { panic!("Unexpected end of line") }
            '"' => { return value }
            _ => { value.push(ch); }
        }
    }

    value
}

fn skip_comment(iter: &mut Peekable<Chars>) {
    while let Some(ch) = iter.next() {
        if ch == '\n' || ch == '\r' {
            return;
        }
    }
}

fn skip_whitespace(iter: &mut Peekable<Chars>) -> Option<char> {
    while let Some(ch) = iter.next() {
        if ! ch.is_whitespace() {
            return Some(ch);
        }
    }

    None
}

fn skip_space(iter: &mut Peekable<Chars>) -> Option<char> {
    while let Some(ch) = iter.next() {
        if ch != ' ' && ch != '\t' {
            return Some(ch);
        }
    }

    None
}

pub struct Config {
    map: HashMap<String, String>,
}

impl Config {
    fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    pub(crate) fn into_arc(self) -> ConfigArc {
        ConfigArc(Arc::new(self))
    }

    fn add_value(&mut self, name: String, value: String) {
        let value = value.trim(); // TODO: in-place trim

        self.map.insert(name, value.to_string());
    }

    pub fn get_with_prefix(&self, prefix: &str, name: &str) -> Option<&String> {
        let name = vec![prefix, name].join(".");

        self.get(&name)
    }

    pub fn get_as_type<T: FromStr>(&self, prefix: &str, name: &str) -> Option<T>
    where <T as FromStr>::Err : fmt::Debug
    {
        match self.get_with_prefix(prefix, name) {
            Some(value) => Some(value.parse::<T>().unwrap()),
            None => None,
        }
    }
    
    pub fn join(&self, prefix: &str, suffix: &str) -> String {
        vec![prefix, suffix].join(".").to_string()
    }

    pub fn get(&self, name: &str) -> Option<&String> {
        if let Some(value) = self.map.get(name) {
            return Some(value);
        }

        let mut split : Vec<&str> = name.split('.').collect();
        while split.len() > 0 {
            split.remove(0);

            let subname = split.join(".");

            if let Some(value) = self.map.get(&subname) {
                return Some(value);
            }
        }

        None
    }
}

#[derive(Clone)]
pub struct ConfigArc(Arc<Config>);

impl Deref for ConfigArc {
    type Target = Config;

    fn deref(&self) -> &Self::Target {
        self.0.deref()
    }
}

#[cfg(test)]
mod test {
    use super::read_config;

    #[test]
    fn config_basic() {
        let config = read_config();

        assert_eq!(config.get("bogus"), None);
        assert_eq!(config.get("grid"), None);
        assert_eq!(config.get("line_width"), None);
        assert_eq!(config.get("grid.line_width"), Some(&"0.8".to_string()));
        assert_eq!(config.get("major.grid.line_width"), Some(&"0.8".to_string()));
        assert_eq!(config.get_with_prefix("major.grid", "line_width"), Some(&"0.8".to_string()));
    }

    #[test]
    fn config_escaped_value() {
        let config = read_config();

        // grid.color is a known escaped value because of "#b0b0b0"
        assert_eq!(config.get("grid.color"), Some(&"#b0b0b0".to_string()));
    }
}