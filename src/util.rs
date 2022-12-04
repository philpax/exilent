use anyhow::Context;
use serenity::{
    async_trait,
    http::Http,
    model::{
        prelude::{
            interaction::{
                application_command::{ApplicationCommandInteraction, CommandDataOptionValue},
                message_component::MessageComponentInteraction,
                modal::ModalSubmitInteraction,
                InteractionResponseType,
            },
            ChannelId, Message,
        },
        user::User,
    },
};

use crate::{constant, sd};

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

pub fn value_to_attachment_url(v: &CommandDataOptionValue) -> Option<String> {
    match v {
        CommandDataOptionValue::Attachment(v) => Some(v.url.clone()),
        _ => None,
    }
}

pub fn get_image_url(aci: &ApplicationCommandInteraction) -> anyhow::Result<String> {
    let image_attachment_url =
        get_value(aci, constant::value::IMAGE_ATTACHMENT).and_then(value_to_attachment_url);

    let image_url = get_value(aci, constant::value::IMAGE_URL).and_then(value_to_string);

    Ok(image_attachment_url
        .or(image_url)
        .context("expected an image to be passed in")?)
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

pub fn encode_image_to_png_bytes(image: image::DynamicImage) -> anyhow::Result<Vec<u8>> {
    let mut bytes: Vec<u8> = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut bytes);
    image.write_to(&mut cursor, image::ImageOutputFormat::Png)?;
    Ok(bytes)
}

#[async_trait]
pub trait DiscordInteraction: Send + Sync {
    async fn create(&self, http: &Http, message: &str) -> anyhow::Result<()>;
    async fn get_interaction_message(&self, http: &Http) -> anyhow::Result<Message>;

    fn channel_id(&self) -> ChannelId;
    fn message(&self) -> Option<&Message>;
    fn user(&self) -> &User;
}
macro_rules! implement_interaction {
    ($name:ident) => {
        #[async_trait]
        impl DiscordInteraction for $name {
            async fn create(&self, http: &Http, msg: &str) -> anyhow::Result<()> {
                Ok(self
                    .create_interaction_response(http, |response| {
                        response
                            .kind(InteractionResponseType::ChannelMessageWithSource)
                            .interaction_response_data(|message| message.content(msg))
                    })
                    .await?)
            }
            async fn get_interaction_message(&self, http: &Http) -> anyhow::Result<Message> {
                Ok(self.get_interaction_response(http).await?)
            }

            fn channel_id(&self) -> ChannelId {
                self.channel_id
            }
            fn user(&self) -> &User {
                &self.user
            }
            interaction_message!($name);
        }
    };
}
macro_rules! interaction_message {
    (ApplicationCommandInteraction) => {
        fn message(&self) -> Option<&Message> {
            None
        }
    };
    (MessageComponentInteraction) => {
        fn message(&self) -> Option<&Message> {
            Some(&self.message)
        }
    };
    (ModalSubmitInteraction) => {
        fn message(&self) -> Option<&Message> {
            self.message.as_ref()
        }
    };
}
implement_interaction!(ApplicationCommandInteraction);
implement_interaction!(MessageComponentInteraction);
implement_interaction!(ModalSubmitInteraction);
