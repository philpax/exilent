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

use crate::{constant, sd, store::Store};

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

pub struct OwnedBaseGenerationParameters<'a> {
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub seed: Option<i64>,
    pub batch_count: Option<u32>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub cfg_scale: Option<f32>,
    pub denoising_strength: Option<f32>,
    pub steps: Option<u32>,
    pub tiling: Option<bool>,
    pub restore_faces: Option<bool>,
    pub sampler: Option<sd::Sampler>,
    pub model: Option<&'a sd::Model>,
}
impl OwnedBaseGenerationParameters<'_> {
    pub fn load<'a>(
        aci: &ApplicationCommandInteraction,
        store: &'a Store,
        models: &'a [sd::Model],
        use_last_generation_for_size: bool,
    ) -> anyhow::Result<OwnedBaseGenerationParameters<'a>> {
        let prompt = get_value(aci, constant::value::PROMPT)
            .and_then(value_to_string)
            .context("expected prompt")?;

        let negative_prompt =
            get_value(aci, constant::value::NEGATIVE_PROMPT).and_then(value_to_string);

        let seed = get_value(aci, constant::value::SEED).and_then(value_to_int);

        let batch_count = get_value(aci, constant::value::COUNT)
            .and_then(value_to_int)
            .map(|v| v as u32);

        let last_generation = store.get_last_generation_for_user(aci.user.id)?;
        let last_generation = last_generation.as_ref();

        let mut width = get_value(aci, constant::value::WIDTH)
            .and_then(value_to_int)
            .map(|v| v as u32 / 64 * 64);

        let mut height = get_value(aci, constant::value::HEIGHT)
            .and_then(value_to_int)
            .map(|v| v as u32 / 64 * 64);

        if use_last_generation_for_size {
            width = width.or_else(|| last_generation.map(|g| g.width));
            height = height.or_else(|| last_generation.map(|g| g.height));
        }

        let cfg_scale = get_value(aci, constant::value::GUIDANCE_SCALE)
            .and_then(value_to_number)
            .map(|v| v as f32)
            .or_else(|| last_generation.map(|g| g.cfg_scale));

        let denoising_strength = get_value(aci, constant::value::DENOISING_STRENGTH)
            .and_then(value_to_number)
            .map(|v| v as f32)
            .or_else(|| last_generation.map(|g| g.denoising_strength));

        let steps = get_value(aci, constant::value::STEPS)
            .and_then(value_to_int)
            .map(|v| v as u32)
            .or_else(|| last_generation.map(|g| g.steps));

        let tiling = get_value(aci, constant::value::TILING)
            .and_then(value_to_bool)
            .or_else(|| last_generation.map(|g| g.tiling));

        let restore_faces = get_value(aci, constant::value::RESTORE_FACES)
            .and_then(value_to_bool)
            .or_else(|| last_generation.map(|g| g.restore_faces));

        let sampler = get_value(aci, constant::value::SAMPLER)
            .and_then(value_to_string)
            .and_then(|v| sd::Sampler::try_from(v.as_str()).ok())
            .or_else(|| last_generation.map(|g| g.sampler));

        let model = {
            let model_params: Vec<_> = get_values_starting_with(aci, constant::value::MODEL)
                .flat_map(value_to_string)
                .collect();
            if model_params.len() > 1 {
                anyhow::bail!("more than one model specified: {:?}", model_params);
            }

            model_params
                .first()
                .and_then(|v| models.iter().find(|m| m.title == *v))
                .or_else(|| {
                    last_generation
                        .and_then(|g| find_model_by_hash(&models, &g.model_hash).map(|t| t.1))
                })
        };

        Ok(OwnedBaseGenerationParameters {
            prompt,
            negative_prompt,
            seed,
            batch_count,
            width,
            height,
            cfg_scale,
            denoising_strength,
            steps,
            tiling,
            restore_faces,
            sampler,
            model,
        })
    }

    pub fn as_base_generation_request(&self) -> sd::BaseGenerationRequest {
        sd::BaseGenerationRequest {
            prompt: self.prompt.as_str(),
            negative_prompt: self.negative_prompt.as_deref(),
            seed: self.seed,
            batch_size: Some(1),
            batch_count: self.batch_count,
            width: self.width,
            height: self.height,
            cfg_scale: self.cfg_scale,
            denoising_strength: self.denoising_strength,
            steps: self.steps,
            tiling: self.tiling,
            restore_faces: self.restore_faces,
            sampler: self.sampler,
            model: self.model,
            ..Default::default()
        }
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
