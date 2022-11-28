use serenity::model::prelude::interaction::application_command::{
    ApplicationCommandInteraction, CommandDataOptionValue,
};

use crate::sd;

pub fn get_value<'a>(
    cmd: &'a ApplicationCommandInteraction,
    name: &'a str,
) -> Option<&'a CommandDataOptionValue> {
    cmd.data
        .options
        .iter()
        .find(|v| v.name == name)
        .and_then(|v| v.resolved.as_ref())
}

pub fn value_to_int(v: Option<&CommandDataOptionValue>) -> Option<i64> {
    match v? {
        CommandDataOptionValue::Integer(v) => Some(*v),
        _ => None,
    }
}

pub fn value_to_number(v: Option<&CommandDataOptionValue>) -> Option<f64> {
    match v? {
        CommandDataOptionValue::Number(v) => Some(*v),
        _ => None,
    }
}

pub fn value_to_string(v: Option<&CommandDataOptionValue>) -> Option<String> {
    match v? {
        CommandDataOptionValue::String(v) => Some(v.clone()),
        _ => None,
    }
}

pub fn value_to_bool(v: Option<&CommandDataOptionValue>) -> Option<bool> {
    match v? {
        CommandDataOptionValue::Boolean(v) => Some(*v),
        _ => None,
    }
}

pub fn generate_chunked_strings(
    strings: impl Iterator<Item = String>,
    threshold: usize,
) -> Vec<String> {
    let mut texts = vec![String::new()];
    for string in strings {
        if texts.last().map(|t| t.len()) >= Some(threshold) {
            texts.push(String::new());
        }
        if let Some(last) = texts.last_mut() {
            if !last.is_empty() {
                *last += ", ";
            }
            *last += &string;
        }
    }
    texts
}

pub fn find_model_by_hash<'a>(models: &'a [sd::Model], model_hash: &str) -> Option<&'a sd::Model> {
    models.iter().find(|m| {
        let Some(hash_wrapped) = m.title.split_ascii_whitespace().last() else { return false };
        &hash_wrapped[1..hash_wrapped.len() - 1] == model_hash
    })
}