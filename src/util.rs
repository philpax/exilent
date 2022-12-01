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

pub fn get_values_starting_with<'a>(
    cmd: &'a ApplicationCommandInteraction,
    name: &'a str,
) -> impl Iterator<Item = &'a CommandDataOptionValue> {
    cmd.data
        .options
        .iter()
        .filter(move |v| v.name.starts_with(name))
        .flat_map(|v| v.resolved.as_ref())
}

pub fn value_to_int(v: &CommandDataOptionValue) -> Option<i64> {
    match v {
        CommandDataOptionValue::Integer(v) => Some(*v),
        _ => None,
    }
}

pub fn value_to_number(v: &CommandDataOptionValue) -> Option<f64> {
    match v {
        CommandDataOptionValue::Number(v) => Some(*v),
        _ => None,
    }
}

pub fn value_to_string(v: &CommandDataOptionValue) -> Option<String> {
    match v {
        CommandDataOptionValue::String(v) => Some(v.clone()),
        _ => None,
    }
}

pub fn value_to_bool(v: &CommandDataOptionValue) -> Option<bool> {
    match v {
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

pub fn find_model_by_hash<'a>(
    models: &'a [sd::Model],
    model_hash: &str,
) -> Option<(usize, &'a sd::Model)> {
    models.iter().enumerate().find(|(_, m)| {
        let Some(hash_wrapped) = m.title.split_ascii_whitespace().last() else { return false };
        &hash_wrapped[1..hash_wrapped.len() - 1] == model_hash
    })
}
