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

    image_attachment_url
        .or(image_url)
        .context("expected an image to be passed in")
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

pub fn find_model_by_hash(models: &[sd::Model], model_hash: &str) -> Option<(usize, sd::Model)> {
    models
        .iter()
        .enumerate()
        .find(|(_, m)| {
            let Some(hash_wrapped) = m.title.split_ascii_whitespace().last() else { return false };
            &hash_wrapped[1..hash_wrapped.len() - 1] == model_hash
        })
        .map(|(idx, model)| (idx, model.clone()))
}

pub fn encode_image_to_png_bytes(image: image::DynamicImage) -> anyhow::Result<Vec<u8>> {
    let mut bytes: Vec<u8> = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut bytes);
    image.write_to(&mut cursor, image::ImageOutputFormat::Png)?;
    Ok(bytes)
}

pub fn fixup_resolution(width: u32, height: u32) -> (u32, u32) {
    use constant::limits::{HEIGHT_MAX, WIDTH_MAX};

    let mut width = width;
    let mut height = height;
    const ROUND_PRECISION: u32 = 64;

    if width > WIDTH_MAX {
        let scale_factor = (width as f32) / (WIDTH_MAX as f32);
        width = ((width as f32) / scale_factor) as u32;
        height = ((height as f32) / scale_factor) as u32;
    }

    if height > HEIGHT_MAX {
        let scale_factor = (height as f32) / (HEIGHT_MAX as f32);
        width = ((width as f32) / scale_factor) as u32;
        height = ((height as f32) / scale_factor) as u32;
    }

    (
        ((width + ROUND_PRECISION / 2) / ROUND_PRECISION) * ROUND_PRECISION,
        ((height + ROUND_PRECISION / 2) / ROUND_PRECISION) * ROUND_PRECISION,
    )
}

pub fn extract_keywords(model_name: &str) -> Vec<&str> {
    let left_index = model_name.rfind('[');
    let right_index = model_name.rfind(']');
    if let Some((left, right)) = left_index.zip(right_index) {
        model_name[left + 1..right]
            .split(',')
            .map(|s| s.trim())
            .collect()
    } else {
        vec![]
    }
}

pub fn prepend_keyword_if_necessary(prompt: &str, model_name: &str) -> String {
    if !constant::config::AUTOMATICALLY_PREPEND_KEYWORD {
        return prompt.to_string();
    }

    prepend_keyword_if_necessary_unchecked(prompt, model_name)
}

fn prepend_keyword_if_necessary_unchecked(prompt: &str, model_name: &str) -> String {
    let keywords = extract_keywords(model_name);
    if let [keyword] = keywords.as_slice() {
        if prompt.contains(keyword) {
            prompt.to_string()
        } else {
            format!("{}, {}", keyword, prompt)
        }
    } else {
        prompt.to_string()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn extract_keyword_works_correctly() {
        use super::extract_keywords;
        assert_eq!(extract_keywords("Inkpunk v1"), Vec::<&str>::new());
        assert_eq!(
            extract_keywords("Inkpunk v1 [nvinkpunk]"),
            vec!["nvinkpunk"]
        );
        assert_eq!(
            extract_keywords("All-In-One Pixel [pixelsprite, 16bitscene]"),
            vec!["pixelsprite", "16bitscene"]
        );
        assert_eq!(
            extract_keywords("All-In-One Pixel [random nonsense] [pixelsprite, 16bitscene]"),
            vec!["pixelsprite", "16bitscene"]
        );
    }

    #[test]
    fn prepend_keyword_if_necessary_unchecked_correctly() {
        use super::prepend_keyword_if_necessary_unchecked;
        assert_eq!(
            prepend_keyword_if_necessary_unchecked("my cool prompt", "Inkpunk v1"),
            "my cool prompt"
        );
        assert_eq!(
            prepend_keyword_if_necessary_unchecked("my cool prompt", "Inkpunk v1 [nvinkpunk]"),
            "nvinkpunk, my cool prompt"
        );
        assert_eq!(
            prepend_keyword_if_necessary_unchecked(
                "my cool prompt",
                "All-In-One Pixel [pixelsprite, 16bitscene]"
            ),
            "my cool prompt"
        );
        assert_eq!(
            prepend_keyword_if_necessary_unchecked(
                "my cool prompt",
                "All-In-One Pixel [random nonsense] [pixelsprite, 16bitscene]"
            ),
            "my cool prompt"
        );
    }
}

#[async_trait]
pub trait DiscordInteraction: Send + Sync {
    async fn create(&self, http: &Http, message: &str) -> anyhow::Result<()>;
    async fn get_interaction_message(&self, http: &Http) -> anyhow::Result<Message>;
    async fn edit(&self, http: &Http, message: &str) -> anyhow::Result<()>;

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
            async fn edit(&self, http: &Http, message: &str) -> anyhow::Result<()> {
                Ok(self
                    .get_interaction_message(http)
                    .await?
                    .edit(http, |m| m.content(message))
                    .await?)
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

macro_rules! create_modal_interaction_response {
    ($title:expr, $custom_id:expr, $generation:ident) => {
        |r| {
            use serenity::model::prelude::component::InputTextStyle;

            r.kind(InteractionResponseType::Modal)
                .interaction_response_data(|d| {
                    d.components(|c| {
                        c.create_action_row(|r| {
                            r.create_input_text(|t| {
                                t.label("Prompt")
                                    .custom_id(constant::value::PROMPT)
                                    .required(true)
                                    .style(InputTextStyle::Short)
                                    .value($generation.prompt)
                            })
                        })
                        .create_action_row(|r| {
                            r.create_input_text(|t| {
                                t.label("Negative prompt")
                                    .custom_id(constant::value::NEGATIVE_PROMPT)
                                    .required(false)
                                    .style(InputTextStyle::Short)
                                    .value($generation.negative_prompt.unwrap_or_default())
                            })
                        })
                        .create_action_row(|r| {
                            r.create_input_text(|t| {
                                t.label("Seed")
                                    .custom_id(constant::value::SEED)
                                    .required(false)
                                    .style(InputTextStyle::Short)
                                    .value($generation.seed)
                            })
                        })
                        .create_action_row(|r| {
                            r.create_input_text(|t| {
                                t.label("Width, height")
                                    .custom_id(constant::value::WIDTH_HEIGHT)
                                    .required(false)
                                    .style(InputTextStyle::Short)
                                    .value(format!("{}, {}", $generation.width, $generation.height))
                            })
                        })
                        .create_action_row(|r| {
                            r.create_input_text(|t| {
                                t.label("Guidance scale, denoising strength")
                                    .custom_id(constant::value::GUIDANCE_SCALE_DENOISING_STRENGTH)
                                    .required(false)
                                    .style(InputTextStyle::Short)
                                    .value(format!(
                                        "{}, {}",
                                        $generation.cfg_scale, $generation.denoising_strength
                                    ))
                            })
                        })
                    })
                    .title($title)
                    .custom_id($custom_id.to_id($generation.id.unwrap()))
                })
        }
    };
}

pub(crate) use create_modal_interaction_response;
