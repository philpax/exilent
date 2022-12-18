use super::issuer;
use crate::{
    config::Configuration,
    constant, custom_id as cid, store,
    util::{self, DiscordInteraction},
};
use anyhow::Context;
use rand::prelude::SliceRandom;
use serenity::{
    http::Http,
    model::prelude::{
        interaction::{
            message_component::MessageComponentInteraction, modal::ModalSubmitInteraction,
            InteractionResponseType,
        },
        *,
    },
};
use stable_diffusion_a1111_webui_client as sd;
use std::{collections::HashMap, str::FromStr};

pub async fn retry(
    client: &sd::Client,
    models: &[sd::Model],
    store: &store::Store,
    http: &Http,
    mci: &MessageComponentInteraction,
    id: i64,
) -> anyhow::Result<()> {
    retry_impl(client, models, store, http, mci, id, Overrides::none(false)).await
}

pub async fn retry_with_options(
    store: &store::Store,
    http: &Http,
    mci: &MessageComponentInteraction,
    id: i64,
) -> anyhow::Result<()> {
    reissue_impl(
        store,
        http,
        mci,
        id,
        "Retry with prompt",
        cid::Generation::RetryWithOptionsResponse,
    )
    .await
}

pub async fn retry_with_options_response(
    client: &sd::Client,
    models: &[sd::Model],
    store: &store::Store,
    http: &Http,
    msi: &ModalSubmitInteraction,
    id: i64,
) -> anyhow::Result<()> {
    reissue_response_impl(client, models, store, http, msi, id, false).await
}

pub async fn remix(
    store: &store::Store,
    http: &Http,
    mci: &MessageComponentInteraction,
    id: i64,
) -> anyhow::Result<()> {
    reissue_impl(
        store,
        http,
        mci,
        id,
        "Remix",
        cid::Generation::RemixResponse,
    )
    .await
}

pub async fn remix_response(
    client: &sd::Client,
    models: &[sd::Model],
    store: &store::Store,
    http: &Http,
    msi: &ModalSubmitInteraction,
    id: i64,
) -> anyhow::Result<()> {
    reissue_response_impl(client, models, store, http, msi, id, true).await
}

pub async fn upscale(
    client: &sd::Client,
    store: &store::Store,
    http: &Http,
    mci: &MessageComponentInteraction,
    id: i64,
) -> anyhow::Result<()> {
    mci.create(http, "Postprocess request received, processing...")
        .await?;

    let (image, url) = store
        .get_generation(id)?
        .map(|g| (g.image, g.image_url))
        .context("generation not found")?;

    let image = image::load_from_memory(&image)?;
    let url = url.as_deref().unwrap_or("unknown");

    mci.edit(http, &format!("Postprocessing {url}...")).await?;

    let result = client
        .postprocess(
            &image,
            &sd::PostprocessRequest {
                resize_mode: sd::ResizeMode::Resize,
                upscaler_1: sd::Upscaler::ESRGAN4x,
                upscaler_2: sd::Upscaler::ESRGAN4x,
                scale_factor: 2.0,
                ..Default::default()
            },
        )
        .await?;

    let bytes = util::encode_image_to_png_bytes(result)?;

    mci.get_interaction_message(http)
        .await?
        .edit(http, |m| {
            m.content(format!("Postprocessing of <{url}> complete."))
                .attachment((bytes.as_slice(), "postprocess.png"))
        })
        .await?;

    Ok(())
}

pub async fn interrogate(
    client: &sd::Client,
    store: &store::Store,
    http: &Http,
    interaction: &dyn DiscordInteraction,
    id: i64,
    interrogator: sd::Interrogator,
) -> anyhow::Result<()> {
    interaction
        .create(http, &format!("Interrogating with {interrogator}..."))
        .await?;

    let image = image::load_from_memory(
        &store
            .get_generation(id)?
            .context("generation not found")?
            .image,
    )?;

    issuer::interrogate_task(
        client,
        store,
        interaction,
        http,
        (
            image,
            store::InterrogationSource::GenerationId(id),
            interrogator,
        ),
    )
    .await
}

pub async fn interrogate_reinterrogate(
    client: &sd::Client,
    store: &store::Store,
    http: &Http,
    interaction: &dyn DiscordInteraction,
    id: i64,
    interrogator: sd::Interrogator,
) -> anyhow::Result<()> {
    interaction
        .create(http, &format!("Interrogating with {interrogator}..."))
        .await?;

    let interrogation = store
        .get_interrogation(id)?
        .context("interrogation not found")?;

    let image = match interrogation.source {
        store::InterrogationSource::GenerationId(id) => {
            store
                .get_generation(id)?
                .context("generation not found")?
                .image
        }
        store::InterrogationSource::Url(url) => reqwest::get(&url).await?.bytes().await?.to_vec(),
    };

    issuer::interrogate_task(
        client,
        store,
        interaction,
        http,
        (
            image::load_from_memory(&image)?,
            store::InterrogationSource::GenerationId(id),
            interrogator,
        ),
    )
    .await
}

pub async fn interrogate_generate(
    client: &sd::Client,
    models: &[sd::Model],
    store: &store::Store,
    http: &Http,
    interaction: &dyn DiscordInteraction,
    id: i64,
) -> anyhow::Result<()> {
    interaction
        .create(
            http,
            "Interrogation generation request received, processing...",
        )
        .await?;
    let interrogation = store
        .get_interrogation(id)?
        .context("no interrogation found")?;

    // use last generation as default if available
    let last_generation = store.get_last_generation_for_user(interaction.user().id)?;
    let last_generation = last_generation.as_ref();

    let base = {
        // always applied
        let prompt = interrogation.result;
        let prompt = if let sd::Interrogator::DeepDanbooru = interrogation.interrogator {
            let mut components: Vec<_> = prompt.split(", ").collect();
            components.shuffle(&mut rand::thread_rng());
            components.join(", ")
        } else {
            prompt
        };

        let width = last_generation.map(|g| g.width);
        let height = last_generation.map(|g| g.height);
        let cfg_scale = last_generation.map(|g| g.cfg_scale);
        let steps = last_generation.map(|g| g.steps);
        let tiling = last_generation.map(|g| g.tiling);
        let restore_faces = last_generation.map(|g| g.restore_faces);
        let sampler = last_generation.map(|g| g.sampler);
        let model = last_generation
            .and_then(|g| util::find_model_by_hash(models, &g.model_hash).map(|t| t.1));

        interaction
            .edit(http, &format!("`{prompt}`: Generating..."))
            .await?;

        let mut base = sd::BaseGenerationRequest {
            prompt: prompt.clone(),
            negative_prompt: None,
            seed: None,
            batch_size: Some(1),
            batch_count: Some(1),
            width,
            height,
            cfg_scale,
            steps,
            tiling,
            restore_faces,
            sampler,
            model,
            ..Default::default()
        };
        util::fixup_base_generation_request(&mut base);
        base
    };
    let prompt = base.prompt.clone();
    issuer::generation_task(
        client.generate_from_text(&sd::TextToImageGenerationRequest {
            base,
            ..Default::default()
        })?,
        models,
        store,
        http,
        interaction,
        None,
        (prompt.as_str(), None),
        None,
    )
    .await
}

async fn reissue_impl(
    store: &store::Store,
    http: &Http,
    mci: &MessageComponentInteraction,
    id: i64,
    title: &str,
    custom_id: cid::Generation,
) -> anyhow::Result<()> {
    let generation = store.get_generation(id)?.context("generation not found")?;

    mci.create_interaction_response(
        http,
        util::create_modal_interaction_response!(title, custom_id, generation),
    )
    .await?;

    Ok(())
}

async fn reissue_response_impl(
    client: &sd::Client,
    models: &[sd::Model],
    store: &store::Store,
    http: &Http,
    msi: &ModalSubmitInteraction,
    id: i64,
    paintover: bool,
) -> anyhow::Result<()> {
    let rows: HashMap<String, String> = msi
        .data
        .components
        .iter()
        .flat_map(|r| r.components.iter())
        .filter_map(|c| {
            if let component::ActionRowComponent::InputText(it) = c {
                Some((it.custom_id.clone(), it.value.clone()))
            } else {
                None
            }
        })
        .collect();

    fn parse_two<T: FromStr, U: FromStr>(value: Option<&String>) -> (Option<T>, Option<U>) {
        fn parse_two_impl<T: FromStr, U: FromStr>(value: Option<&String>) -> Option<(T, U)> {
            fn trim_parse<T: FromStr>(value: &str) -> Option<T> {
                value.trim().parse().ok()
            }

            let (str1, str2) = value?.split_once(',')?;
            Some((trim_parse(str1)?, trim_parse(str2)?))
        }
        match parse_two_impl(value) {
            Some((t, u)) => (Some(t), Some(u)),
            _ => (None, None),
        }
    }

    let prompt = rows.get(constant::value::PROMPT).map(|s| s.as_str());

    let negative_prompt = rows
        .get(constant::value::NEGATIVE_PROMPT)
        .map(|s| s.as_str());

    let (width, height) = parse_two(rows.get(constant::value::WIDTH_HEIGHT));

    let seed = rows
        .get(constant::value::SEED)
        .map(|s| s.parse::<i64>().ok());

    let (guidance_scale, denoising_strength) =
        parse_two(rows.get(constant::value::GUIDANCE_SCALE_DENOISING_STRENGTH));

    retry_impl(
        client,
        models,
        store,
        http,
        msi,
        id,
        Overrides::new(
            prompt,
            negative_prompt,
            width,
            height,
            guidance_scale,
            None,
            seed,
            denoising_strength,
            paintover,
        ),
    )
    .await
}

async fn retry_impl(
    client: &sd::Client,
    models: &[sd::Model],
    store: &store::Store,
    http: &Http,
    interaction: &dyn DiscordInteraction,
    id: i64,
    overrides: Overrides<'_>,
) -> anyhow::Result<()> {
    interaction
        .create(http, "Retry request received, processing...")
        .await?;
    let mut generation = store
        .get_generation(id)?
        .context("generation not found")?
        .clone();

    if overrides.paintover {
        let init_image = image::load_from_memory(&generation.image)?;
        let init_url = generation
            .image_url
            .clone()
            .unwrap_or_else(|| "UNKNOWN".to_string());

        if let Some(image_generation) = generation.image_generation.as_mut() {
            image_generation.init_image = init_image;
            image_generation.init_url = init_url;
        } else {
            generation.image_generation = Some(store::ImageGeneration {
                init_image,
                init_url,
                resize_mode: Default::default(),
            });
        }
    }

    let mut request = generation.as_generation_request(models);
    {
        let base = match &mut request {
            store::GenerationRequest::Text(r) => &mut r.base,
            store::GenerationRequest::Image(r) => &mut r.base,
        };
        if let Some(prompt) = overrides.prompt {
            base.prompt = prompt.to_string();
        }
        if let Some(negative_prompt) = overrides.negative_prompt {
            base.negative_prompt = Some(negative_prompt.to_string());
        }
        if let Some(width) = overrides.width {
            base.width = Some(width);
        }
        if let Some(height) = overrides.height {
            base.height = Some(height);
        }
        if let Some(guidance_scale) = overrides.guidance_scale {
            base.cfg_scale = Some(guidance_scale as f32);
        }
        if let Some(steps) = overrides.steps {
            base.steps = Some(steps as u32);
        }
        if let Some(seed) = overrides.seed {
            base.seed = seed;
        }
        if let Some(denoising_strength) = overrides.denoising_strength {
            base.denoising_strength = Some(denoising_strength as f32);
        }
        util::fixup_base_generation_request(base);
    }
    interaction
        .edit(
            http,
            &format!(
                "`{}`{}: Generating retry...",
                request.base().prompt,
                request
                    .base()
                    .negative_prompt
                    .as_ref()
                    .filter(|s| !s.is_empty())
                    .map(|s| format!(" - `{s}`"))
                    .unwrap_or_default()
            ),
        )
        .await?;

    issuer::generation_task(
        request.generate(client)?,
        models,
        store,
        http,
        interaction,
        None,
        (
            &request.base().prompt,
            request.base().negative_prompt.as_deref(),
        ),
        generation.image_generation.clone(),
    )
    .await?;

    Ok(())
}

struct Overrides<'a> {
    prompt: Option<&'a str>,
    negative_prompt: Option<&'a str>,
    width: Option<u32>,
    height: Option<u32>,
    guidance_scale: Option<f64>,
    steps: Option<usize>,
    /// None: don't override
    /// Some(None): override with a fresh seed
    /// Some(Some(seed)): override with seed
    seed: Option<Option<i64>>,
    denoising_strength: Option<f64>,
    paintover: bool,
}
impl<'a> Overrides<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        prompt: Option<&'a str>,
        negative_prompt: Option<&'a str>,
        width: Option<u32>,
        height: Option<u32>,
        guidance_scale: Option<f64>,
        steps: Option<usize>,
        seed: Option<Option<i64>>,
        denoising_strength: Option<f64>,
        paintover: bool,
    ) -> Self {
        let l = &Configuration::get().limits;

        Self {
            prompt: prompt.filter(|s| !s.is_empty()),
            negative_prompt: negative_prompt,
            width,
            height,
            guidance_scale: guidance_scale
                .map(|s| s.clamp(l.guidance_scale_min, l.guidance_scale_max)),
            steps: steps.map(|s| s.clamp(l.steps_min, l.steps_max)),
            seed,
            denoising_strength: denoising_strength.map(|s| s.clamp(0.0, 1.0)),
            paintover,
        }
    }

    fn none(paintover: bool) -> Self {
        Self {
            prompt: None,
            negative_prompt: None,
            width: None,
            height: None,
            guidance_scale: None,
            steps: None,
            seed: Some(None),
            denoising_strength: None,
            paintover,
        }
    }
}
