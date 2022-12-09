use std::{
    collections::{HashMap, HashSet},
    env,
    str::FromStr,
};

use anyhow::Context as AnyhowContext;
use dotenv::dotenv;
use rand::seq::SliceRandom;
use serenity::{
    async_trait,
    builder::CreateApplicationCommand,
    client::{Context, EventHandler},
    http::Http,
    model::{
        application::interaction::{Interaction, InteractionResponseType},
        prelude::{
            command::{Command, CommandOptionType},
            component::InputTextStyle,
            interaction::{
                application_command::ApplicationCommandInteraction,
                message_component::MessageComponentInteraction, modal::ModalSubmitInteraction,
            },
            *,
        },
    },
    Client,
};

mod constant;
mod custom_id;
mod issuer;
mod store;
mod util;

use custom_id as cid;
use stable_diffusion_a1111_webui_client as sd;
use store::Store;
use util::DiscordInteraction;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv().ok();

    let client = {
        let sd_url = env::var("SD_URL").expect("SD_URL not specified");
        let sd_authentication = env::var("SD_USER").ok().zip(env::var("SD_PASS").ok());
        sd::Client::new(
            &sd_url,
            sd_authentication
                .as_ref()
                .map(|p| sd::Authentication::ApiAuth(&p.0, &p.1))
                .unwrap_or(sd::Authentication::None),
        )
        .await
        .unwrap()
    };
    let models = client.models().await?;
    let store = Store::load()?;
    let safe_tags = include_str!("safe_tags.txt").lines().collect();

    // Build our client.
    let mut client = Client::builder(
        env::var("DISCORD_TOKEN").expect("Expected a token in the environment"),
        GatewayIntents::default(),
    )
    .event_handler(Handler {
        client,
        models,
        store,
        safe_tags,
    })
    .await
    .expect("Error creating client");

    // Finally, start a single shard, and start listening to events.
    // Shards will automatically attempt to reconnect, and will perform
    // exponential backoff until it reconnects.
    if let Err(why) = client.start().await {
        println!("Client error: {:?}", why);
    }

    Ok(())
}

struct Handler {
    client: sd::Client,
    models: Vec<sd::Model>,
    store: Store,
    safe_tags: HashSet<&'static str>,
}
/// Commands
impl Handler {
    async fn exilent(&self, http: &Http, cmd: ApplicationCommandInteraction) -> anyhow::Result<()> {
        let channel = cmd.channel_id;

        match cmd.data.options[0].name.as_str() {
            "embeddings" => {
                cmd.create_interaction_response(http, |response| {
                    response
                        .kind(InteractionResponseType::ChannelMessageWithSource)
                        .interaction_response_data(|message| {
                            message.title("Embeddings").content("Processing...")
                        })
                })
                .await?;
                let texts = match self.client.embeddings().await {
                    Ok(embeddings) => util::generate_chunked_strings(
                        embeddings.iter().map(|s| format!("`{s}`")),
                        1900,
                    ),
                    Err(err) => vec![format!("{err:?}")],
                };
                cmd.edit(http, texts.first().unwrap()).await?;

                for remainder in texts.iter().skip(1) {
                    channel
                        .send_message(http, |msg| msg.content(remainder))
                        .await?;
                }

                Ok(())
            }
            "stats" => {
                let stats = self.store.get_model_usage_counts()?;
                async fn get_user_name(
                    http: &Http,
                    guild_id: Option<GuildId>,
                    id: UserId,
                ) -> anyhow::Result<(UserId, String)> {
                    let user = id.to_user(http).await?;
                    let name = if let Some(guild_id) = guild_id {
                        user.nick_in(http, guild_id).await
                    } else {
                        None
                    };
                    Ok((id, name.unwrap_or(user.name)))
                }
                let users = futures::future::join_all(
                    stats
                        .keys()
                        .map(|id| get_user_name(http, cmd.guild_id, *id)),
                )
                .await
                .into_iter()
                .collect::<Result<HashMap<_, _>, _>>()?;

                let message = stats
                    .into_iter()
                    .flat_map(|(user, counts)| {
                        std::iter::once(format!(
                            "**{}**",
                            users
                                .get(&user)
                                .map(|s| s.as_str())
                                .unwrap_or("unknown user")
                        ))
                        .chain(counts.into_iter().map(
                            |(model_hash, count)| {
                                format!(
                                    "- {}: {} generations",
                                    util::find_model_by_hash(&self.models, &model_hash)
                                        .map(|m| m.1.name.as_str())
                                        .unwrap_or("unknown model"),
                                    count
                                )
                            },
                        ))
                    })
                    .collect::<Vec<String>>()
                    .join("\n");

                cmd.create(http, &message).await?;
                Ok(())
            }
            _ => unreachable!(),
        }
    }

    async fn paint(&self, http: &Http, aci: ApplicationCommandInteraction) -> anyhow::Result<()> {
        let interaction: &dyn DiscordInteraction = &aci;
        interaction
            .create(http, "Paint request received, processing...")
            .await?;
        let mut base_parameters =
            util::OwnedBaseGenerationParameters::load(&aci, &self.store, &self.models, true)?;

        interaction
            .edit(
                http,
                &format!(
                    "`{}`{}: Generating...",
                    &base_parameters.prompt,
                    base_parameters
                        .negative_prompt
                        .as_deref()
                        .map(|s| format!(" - `{s}`"))
                        .unwrap_or_default()
                ),
            )
            .await?;

        if let Some((width, height)) = base_parameters
            .width
            .as_mut()
            .zip(base_parameters.height.as_mut())
        {
            (*width, *height) = util::fixup_resolution(*width, *height);
        }

        issuer::generation_task(
            self.client
                .generate_from_text(&sd::TextToImageGenerationRequest {
                    base: base_parameters.as_base_generation_request(),
                    ..Default::default()
                })?,
            &self.models,
            &self.store,
            http,
            interaction,
            (
                &base_parameters.prompt,
                base_parameters.negative_prompt.as_deref(),
            ),
            None,
        )
        .await
    }

    async fn paintover(
        &self,
        http: &Http,
        aci: ApplicationCommandInteraction,
    ) -> anyhow::Result<()> {
        let interaction: &dyn DiscordInteraction = &aci;
        interaction
            .create(http, "Paintover request received, processing...")
            .await?;
        let mut base_parameters =
            util::OwnedBaseGenerationParameters::load(&aci, &self.store, &self.models, false)?;
        let url = util::get_image_url(&aci)?;
        let resize_mode = util::get_value(&aci, constant::value::RESIZE_MODE)
            .and_then(util::value_to_string)
            .and_then(|s| sd::ResizeMode::try_from(s.as_str()).ok())
            .unwrap_or_default();

        interaction
            .edit(
                http,
                &format!(
                    "`{}`{}: Painting over {}...",
                    &base_parameters.prompt,
                    base_parameters
                        .negative_prompt
                        .as_deref()
                        .map(|s| format!(" - `{s}`"))
                        .unwrap_or_default(),
                    url
                ),
            )
            .await?;

        let bytes = reqwest::get(&url).await?.bytes().await?;
        let image = image::load_from_memory(&bytes)?;

        if base_parameters.width.is_none() {
            base_parameters.width = Some(image.width());
        }
        if base_parameters.height.is_none() {
            base_parameters.height = Some(image.height());
        }

        if let Some((width, height)) = base_parameters
            .width
            .as_mut()
            .zip(base_parameters.height.as_mut())
        {
            (*width, *height) = util::fixup_resolution(*width, *height);
        }

        issuer::generation_task(
            self.client
                .generate_from_image_and_text(&sd::ImageToImageGenerationRequest {
                    base: base_parameters.as_base_generation_request(),
                    images: vec![&image],
                    resize_mode: Some(resize_mode),
                    ..Default::default()
                })?,
            &self.models,
            &self.store,
            http,
            interaction,
            (
                &base_parameters.prompt,
                base_parameters.negative_prompt.as_deref(),
            ),
            Some(store::ImageGeneration {
                init_image: image.clone(),
                init_url: url.clone(),
                resize_mode,
            }),
        )
        .await
    }

    async fn paintagain(
        &self,
        http: &Http,
        aci: ApplicationCommandInteraction,
    ) -> anyhow::Result<()> {
        let generation = self
            .store
            .get_last_generation_for_user(aci.user.id)?
            .context("no last generation for user")?;

        aci.create_interaction_response(
            http,
            util::create_modal_interaction_response!(
                "Paint Again",
                cid::Generation::RetryWithOptionsResponse,
                generation
            ),
        )
        .await?;
        Ok(())
    }

    async fn postprocess(
        &self,
        http: &Http,
        aci: ApplicationCommandInteraction,
    ) -> anyhow::Result<()> {
        let interaction: &dyn DiscordInteraction = &aci;
        interaction
            .create(http, "Postprocess request received, processing...")
            .await?;

        let url = util::get_image_url(&aci)?;

        interaction
            .edit(http, &format!("Postprocessing {url}..."))
            .await?;

        let bytes = reqwest::get(&url).await?.bytes().await?;
        let image = image::load_from_memory(&bytes)?;

        let upscaler_1 = util::get_value(&aci, constant::value::UPSCALER_1)
            .and_then(util::value_to_string)
            .and_then(|v| sd::Upscaler::try_from(v.as_str()).ok())
            .context("expected upscaler 1")?;

        let upscaler_2 = util::get_value(&aci, constant::value::UPSCALER_2)
            .and_then(util::value_to_string)
            .and_then(|v| sd::Upscaler::try_from(v.as_str()).ok())
            .context("expected upscaler 2")?;

        let scale_factor = util::get_value(&aci, constant::value::SCALE_FACTOR)
            .and_then(util::value_to_number)
            .map(|n| n as f32)
            .context("expected scale factor")?;

        let codeformer_visibility = util::get_value(&aci, constant::value::CODEFORMER_VISIBILITY)
            .and_then(util::value_to_number)
            .map(|n| n as f32);

        let codeformer_weight = util::get_value(&aci, constant::value::CODEFORMER_WEIGHT)
            .and_then(util::value_to_number)
            .map(|n| n as f32);

        let upscaler_2_visibility = util::get_value(&aci, constant::value::UPSCALER_2_VISIBILITY)
            .and_then(util::value_to_number)
            .map(|n| n as f32);

        let gfpgan_visibility = util::get_value(&aci, constant::value::GFPGAN_VISIBILITY)
            .and_then(util::value_to_number)
            .map(|n| n as f32);

        let upscale_first =
            util::get_value(&aci, constant::value::UPSCALE_FIRST).and_then(util::value_to_bool);

        let result = self
            .client
            .postprocess(
                &image,
                &sd::PostprocessRequest {
                    resize_mode: sd::ResizeMode::Resize,
                    upscaler_1,
                    upscaler_2,
                    scale_factor,
                    codeformer_visibility,
                    codeformer_weight,
                    upscaler_2_visibility,
                    gfpgan_visibility,
                    upscale_first,
                },
            )
            .await?;

        let bytes = util::encode_image_to_png_bytes(result)?;

        interaction
            .get_interaction_message(http)
            .await?
            .edit(http, |m| {
                m.content(format!("Postprocessing of <{url}> complete."))
                    .attachment((bytes.as_slice(), "postprocess.png"))
            })
            .await?;

        Ok(())
    }

    async fn interrogate(
        &self,
        http: &Http,
        aci: ApplicationCommandInteraction,
    ) -> anyhow::Result<()> {
        let url = util::get_image_url(&aci)?;

        let interrogator = util::get_value(&aci, constant::value::INTERROGATOR)
            .and_then(util::value_to_string)
            .and_then(|v| sd::Interrogator::try_from(v.as_str()).ok())
            .context("expected interrogator")?;

        let interaction: &dyn DiscordInteraction = &aci;
        interaction
            .create(http, &format!("Interrogating {url} with {interrogator}..."))
            .await?;

        let bytes = reqwest::get(&url).await?.bytes().await?;
        let image = image::load_from_memory(&bytes)?;

        issuer::interrogate_task(
            &self.client,
            &self.store,
            &self.safe_tags,
            interaction,
            http,
            (image, store::InterrogationSource::Url(url), interrogator),
        )
        .await
    }

    async fn png_info(
        &self,
        http: &Http,
        aci: ApplicationCommandInteraction,
    ) -> anyhow::Result<()> {
        let url = util::get_image_url(&aci)?;

        let interaction: &dyn DiscordInteraction = &aci;
        interaction
            .create(http, &format!("Reading PNG info of {url}..."))
            .await?;

        let bytes = reqwest::get(&url).await?.bytes().await?;
        let result = self.client.png_info(&bytes).await?;
        interaction.edit(http, &result).await?;

        Ok(())
    }
}

/// Message component interactions
impl Handler {
    async fn mc_retry(
        &self,
        http: &Http,
        mci: &MessageComponentInteraction,
        id: i64,
    ) -> anyhow::Result<()> {
        self.mc_retry_impl(http, mci, id, Overrides::none(), false)
            .await
    }

    async fn mc_retry_with_options(
        &self,
        http: &Http,
        mci: &MessageComponentInteraction,
        id: i64,
    ) -> anyhow::Result<()> {
        self.mc_reissue_impl(
            http,
            mci,
            id,
            "Retry with prompt",
            cid::Generation::RetryWithOptionsResponse,
        )
        .await
    }

    async fn mc_retry_with_options_response(
        &self,
        http: &Http,
        msi: &ModalSubmitInteraction,
        id: i64,
    ) -> anyhow::Result<()> {
        self.mc_reissue_response_impl(http, msi, id, false).await
    }

    async fn mc_remix(
        &self,
        http: &Http,
        mci: &MessageComponentInteraction,
        id: i64,
    ) -> anyhow::Result<()> {
        self.mc_reissue_impl(http, mci, id, "Remix", cid::Generation::RemixResponse)
            .await
    }

    async fn mc_remix_response(
        &self,
        http: &Http,
        msi: &ModalSubmitInteraction,
        id: i64,
    ) -> anyhow::Result<()> {
        self.mc_reissue_response_impl(http, msi, id, true).await
    }

    async fn mc_upscale(
        &self,
        http: &Http,
        mci: &MessageComponentInteraction,
        id: i64,
    ) -> anyhow::Result<()> {
        mci.create(http, "Postprocess request received, processing...")
            .await?;

        let (image, url) = self
            .store
            .get_generation(id)?
            .map(|g| (g.image, g.image_url))
            .context("generation not found")?;

        let image = image::load_from_memory(&image)?;
        let url = url.as_deref().unwrap_or("unknown");

        mci.edit(http, &format!("Postprocessing {url}...")).await?;

        let result = self
            .client
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

    async fn mc_reissue_impl(
        &self,
        http: &Http,
        mci: &MessageComponentInteraction,
        id: i64,
        title: &str,
        custom_id: cid::Generation,
    ) -> anyhow::Result<()> {
        let generation = self
            .store
            .get_generation(id)?
            .context("generation not found")?;

        mci.create_interaction_response(
            http,
            util::create_modal_interaction_response!(title, custom_id, generation),
        )
        .await?;

        Ok(())
    }

    async fn mc_reissue_response_impl(
        &self,
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

        self.mc_retry_impl(
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
            ),
            paintover,
        )
        .await
    }

    async fn mc_retry_impl(
        &self,
        http: &Http,
        interaction: &dyn DiscordInteraction,
        id: i64,
        overrides: Overrides<'_>,
        paintover: bool,
    ) -> anyhow::Result<()> {
        interaction
            .create(http, "Retry request received, processing...")
            .await?;
        let mut generation = self
            .store
            .get_generation(id)?
            .context("generation not found")?
            .clone();

        if paintover {
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

        let mut request = generation.as_generation_request(&self.models);
        {
            let base = match &mut request {
                store::GenerationRequest::Text(r) => &mut r.base,
                store::GenerationRequest::Image(r) => &mut r.base,
            };
            if let Some(prompt) = overrides.prompt {
                base.prompt = prompt;
            }
            if let Some(negative_prompt) = overrides.negative_prompt {
                base.negative_prompt = Some(negative_prompt).filter(|s| !s.is_empty());
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
                        .map(|s| format!(" - `{s}`"))
                        .unwrap_or_default()
                ),
            )
            .await?;

        issuer::generation_task(
            request.generate(&self.client)?,
            &self.models,
            &self.store,
            http,
            interaction,
            (
                &request.base().prompt,
                request.base().negative_prompt.as_deref(),
            ),
            generation.image_generation.clone(),
        )
        .await?;

        Ok(())
    }

    async fn mc_interrogate(
        &self,
        http: &Http,
        interaction: &dyn DiscordInteraction,
        id: i64,
        interrogator: sd::Interrogator,
    ) -> anyhow::Result<()> {
        interaction
            .create(http, &format!("Interrogating with {interrogator}..."))
            .await?;

        let image = image::load_from_memory(
            &self
                .store
                .get_generation(id)?
                .context("generation not found")?
                .image,
        )?;

        issuer::interrogate_task(
            &self.client,
            &self.store,
            &self.safe_tags,
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

    async fn mc_interrogate_reinterrogate(
        &self,
        http: &Http,
        interaction: &dyn DiscordInteraction,
        id: i64,
        interrogator: sd::Interrogator,
    ) -> anyhow::Result<()> {
        interaction
            .create(http, &format!("Interrogating with {interrogator}..."))
            .await?;

        let interrogation = self
            .store
            .get_interrogation(id)?
            .context("interrogation not found")?;

        let image = match interrogation.source {
            store::InterrogationSource::GenerationId(id) => {
                self.store
                    .get_generation(id)?
                    .context("generation not found")?
                    .image
            }
            store::InterrogationSource::Url(url) => {
                reqwest::get(&url).await?.bytes().await?.to_vec()
            }
        };

        issuer::interrogate_task(
            &self.client,
            &self.store,
            &self.safe_tags,
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

    async fn mc_interrogate_generate(
        &self,
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
        let interrogation = self
            .store
            .get_interrogation(id)?
            .context("no interrogation found")?;

        // always applied
        let prompt = interrogation.result;
        let prompt = if let sd::Interrogator::DeepDanbooru = interrogation.interrogator {
            let mut components: Vec<_> = prompt.split(", ").collect();
            components.shuffle(&mut rand::thread_rng());
            components.join(", ")
        } else {
            prompt
        };

        // use last generation as default if available
        let last_generation = self
            .store
            .get_last_generation_for_user(interaction.user().id)?;
        let last_generation = last_generation.as_ref();

        let width = last_generation.map(|g| g.width);
        let height = last_generation.map(|g| g.height);
        let cfg_scale = last_generation.map(|g| g.cfg_scale);
        let steps = last_generation.map(|g| g.steps);
        let tiling = last_generation.map(|g| g.tiling);
        let restore_faces = last_generation.map(|g| g.restore_faces);
        let sampler = last_generation.map(|g| g.sampler);
        let model = last_generation
            .and_then(|g| util::find_model_by_hash(&self.models, &g.model_hash).map(|t| t.1));

        interaction
            .edit(http, &format!("`{prompt}`: Generating..."))
            .await?;

        issuer::generation_task(
            self.client
                .generate_from_text(&sd::TextToImageGenerationRequest {
                    base: sd::BaseGenerationRequest {
                        prompt: prompt.as_str(),
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
                    },
                    ..Default::default()
                })?,
            &self.models,
            &self.store,
            http,
            interaction,
            (&prompt, None),
            None,
        )
        .await
    }
}

#[async_trait]
impl EventHandler for Handler {
    async fn ready(&self, ctx: Context, ready: Ready) {
        println!("{} is connected!", ready.user.name);

        Command::create_global_application_command(&ctx.http, |command| {
            command
                .name(constant::command::PAINT)
                .description("Paints your dreams");

            populate_generate_options(command, &self.models);
            command
        })
        .await
        .unwrap();

        Command::create_global_application_command(&ctx.http, |command| {
            command
                .name(constant::command::PAINTOVER)
                .description("Paints your dreams over another image");

            populate_generate_options(command, &self.models);
            command
                .create_option(|option| {
                    option
                        .name(constant::value::IMAGE_URL)
                        .description("The URL of the image to paint over")
                        .kind(CommandOptionType::String)
                })
                .create_option(|option| {
                    option
                        .name(constant::value::IMAGE_ATTACHMENT)
                        .description("The image to paint over")
                        .kind(CommandOptionType::Attachment)
                })
                .create_option(|option| {
                    let opt = option
                        .name(constant::value::RESIZE_MODE)
                        .description("How to resize the image to match the generation")
                        .kind(CommandOptionType::String);

                    for value in sd::ResizeMode::VALUES {
                        opt.add_string_choice(value, value);
                    }

                    opt
                })
        })
        .await
        .unwrap();

        Command::create_global_application_command(&ctx.http, |command| {
            command
                .name(constant::command::PAINTAGAIN)
                .description("Re-run the last generation command with a modal of overrides")
        })
        .await
        .unwrap();

        Command::create_global_application_command(&ctx.http, |command| {
            command
                .name(constant::command::POSTPROCESS)
                .description("Postprocesses an image");

            command
                .create_option(|option| {
                    let opt = option
                        .name(constant::value::UPSCALER_1)
                        .description("The first upscaler")
                        .kind(CommandOptionType::String)
                        .required(true);

                    for value in sd::Upscaler::VALUES {
                        opt.add_string_choice(value, value);
                    }

                    opt
                })
                .create_option(|option| {
                    let opt = option
                        .name(constant::value::UPSCALER_2)
                        .description("The second upscaler")
                        .kind(CommandOptionType::String)
                        .required(true);

                    for value in sd::Upscaler::VALUES {
                        opt.add_string_choice(value, value);
                    }

                    opt
                })
                .create_option(|option| {
                    option
                        .name(constant::value::SCALE_FACTOR)
                        .description("The factor by which to upscale the image")
                        .kind(CommandOptionType::Number)
                        .min_number_value(1.0)
                        .max_number_value(3.0)
                        .required(true)
                })
                .create_option(|option| {
                    option
                        .name(constant::value::IMAGE_URL)
                        .description("The URL of the image to paint over")
                        .kind(CommandOptionType::String)
                })
                .create_option(|option| {
                    option
                        .name(constant::value::IMAGE_ATTACHMENT)
                        .description("The image to paint over")
                        .kind(CommandOptionType::Attachment)
                })
                .create_option(|option| {
                    option
                        .name(constant::value::CODEFORMER_VISIBILITY)
                        .description("How much of CodeFormer's result is blended into the result?")
                        .kind(CommandOptionType::Number)
                        .min_number_value(0.0)
                        .max_number_value(1.0)
                })
                .create_option(|option| {
                    option
                        .name(constant::value::CODEFORMER_WEIGHT)
                        .description("How strong is CodeFormer's effect?")
                        .kind(CommandOptionType::Number)
                        .min_number_value(0.0)
                        .max_number_value(1.0)
                })
                .create_option(|option| {
                    option
                        .name(constant::value::UPSCALER_2_VISIBILITY)
                        .description(
                            "How much of the second upscaler's result is blended into the result?",
                        )
                        .kind(CommandOptionType::Number)
                        .min_number_value(0.0)
                        .max_number_value(1.0)
                })
                .create_option(|option| {
                    option
                        .name(constant::value::GFPGAN_VISIBILITY)
                        .description("How much of GFPGAN's result is blended into the result?")
                        .kind(CommandOptionType::Number)
                        .min_number_value(0.0)
                        .max_number_value(2.0)
                })
                .create_option(|option| {
                    option
                        .name(constant::value::UPSCALE_FIRST)
                        .description("Should the upscaler be applied before other postprocessing?")
                        .kind(CommandOptionType::Boolean)
                })
        })
        .await
        .unwrap();

        Command::create_global_application_command(&ctx.http, |command| {
            command
                .name(constant::command::INTERROGATE)
                .description("Interrogates an image to produce a caption")
                .create_option(|option| {
                    let opt = option
                        .name(constant::value::INTERROGATOR)
                        .description("The interrogator to use")
                        .kind(CommandOptionType::String)
                        .required(true);

                    for value in sd::Interrogator::VALUES {
                        opt.add_string_choice(value, value);
                    }

                    opt
                })
                .create_option(|option| {
                    option
                        .name(constant::value::IMAGE_URL)
                        .description("The URL of the image to interrogate")
                        .kind(CommandOptionType::String)
                })
                .create_option(|option| {
                    option
                        .name(constant::value::IMAGE_ATTACHMENT)
                        .description("The image to interrogate")
                        .kind(CommandOptionType::Attachment)
                })
        })
        .await
        .unwrap();

        Command::create_global_application_command(&ctx.http, |command| {
            command
                .name(constant::command::EXILENT)
                .description("Meta-commands for Exilent")
                .create_option(|option| {
                    option
                        .name("embeddings")
                        .description("Lists all of the supported embeddings")
                        .kind(CommandOptionType::SubCommand)
                })
                .create_option(|option| {
                    option
                        .name("stats")
                        .description("Output some statistics")
                        .kind(CommandOptionType::SubCommand)
                })
        })
        .await
        .unwrap();

        Command::create_global_application_command(&ctx.http, |command| {
            command
                .name(constant::command::PNG_INFO)
                .description("Retrieves the embedded PNG info of an image")
                .create_option(|option| {
                    option
                        .name(constant::value::IMAGE_URL)
                        .description("The URL of the image to read")
                        .kind(CommandOptionType::String)
                })
                .create_option(|option| {
                    option
                        .name(constant::value::IMAGE_ATTACHMENT)
                        .description("The image to read")
                        .kind(CommandOptionType::Attachment)
                })
        })
        .await
        .unwrap();
    }

    async fn interaction_create(&self, ctx: Context, interaction: Interaction) {
        let mut channel_id = None;
        let http = &ctx.http;
        let result = match interaction {
            Interaction::ApplicationCommand(cmd) => {
                channel_id = Some(cmd.channel_id);
                match cmd.data.name.as_str() {
                    constant::command::PAINT => self.paint(http, cmd).await,
                    constant::command::PAINTOVER => self.paintover(http, cmd).await,
                    constant::command::PAINTAGAIN => self.paintagain(http, cmd).await,
                    constant::command::POSTPROCESS => self.postprocess(http, cmd).await,
                    constant::command::INTERROGATE => self.interrogate(http, cmd).await,
                    constant::command::EXILENT => self.exilent(http, cmd).await,
                    constant::command::PNG_INFO => self.png_info(http, cmd).await,
                    _ => Ok(()),
                }
            }
            Interaction::MessageComponent(mci) => {
                channel_id = Some(mci.channel_id);

                let custom_id = cid::CustomId::try_from(mci.data.custom_id.as_str())
                    .expect("invalid interaction id");

                match custom_id {
                    cid::CustomId::Generation { id, value } => match value {
                        cid::Generation::Retry => self.mc_retry(http, &mci, id).await,
                        cid::Generation::RetryWithOptions => {
                            self.mc_retry_with_options(http, &mci, id).await
                        }
                        cid::Generation::Remix => self.mc_remix(http, &mci, id).await,
                        cid::Generation::Upscale => self.mc_upscale(http, &mci, id).await,
                        cid::Generation::InterrogateClip => {
                            self.mc_interrogate(http, &mci, id, sd::Interrogator::Clip)
                                .await
                        }
                        cid::Generation::InterrogateDeepDanbooru => {
                            self.mc_interrogate(http, &mci, id, sd::Interrogator::DeepDanbooru)
                                .await
                        }
                        cid::Generation::RetryWithOptionsResponse => unreachable!(),
                        cid::Generation::RemixResponse => unreachable!(),
                    },
                    cid::CustomId::Interrogation { id, value } => match value {
                        cid::Interrogation::Generate => {
                            self.mc_interrogate_generate(http, &mci, id).await
                        }
                        cid::Interrogation::ReinterrogateWithClip => {
                            self.mc_interrogate_reinterrogate(
                                http,
                                &mci,
                                id,
                                sd::Interrogator::Clip,
                            )
                            .await
                        }
                        cid::Interrogation::ReinterrogateWithDeepDanbooru => {
                            self.mc_interrogate_reinterrogate(
                                http,
                                &mci,
                                id,
                                sd::Interrogator::DeepDanbooru,
                            )
                            .await
                        }
                    },
                }
            }
            Interaction::ModalSubmit(msi) => {
                channel_id = Some(msi.channel_id);

                let custom_id = cid::CustomId::try_from(msi.data.custom_id.as_str())
                    .expect("invalid interaction id");

                match custom_id {
                    cid::CustomId::Generation { id, value } => match value {
                        cid::Generation::RetryWithOptionsResponse => {
                            self.mc_retry_with_options_response(http, &msi, id).await
                        }
                        cid::Generation::RemixResponse => {
                            self.mc_remix_response(http, &msi, id).await
                        }

                        cid::Generation::Retry => unreachable!(),
                        cid::Generation::RetryWithOptions => unreachable!(),
                        cid::Generation::Remix => unreachable!(),
                        cid::Generation::Upscale => unreachable!(),
                        cid::Generation::InterrogateClip => unreachable!(),
                        cid::Generation::InterrogateDeepDanbooru => unreachable!(),
                    },
                    cid::CustomId::Interrogation { .. } => todo!(),
                }
            }
            _ => Ok(()),
        };
        if let Some(channel_id) = channel_id {
            if let Err(err) = result.as_ref() {
                channel_id
                    .send_message(http, |r| r.content(format!("Error: {err:?}")))
                    .await
                    .ok();

                result.unwrap();
            }
        }
    }
}

fn populate_generate_options(command: &mut CreateApplicationCommand, models: &[sd::Model]) {
    command
        .create_option(|option| {
            option
                .name(constant::value::PROMPT)
                .description("The prompt to draw")
                .kind(CommandOptionType::String)
                .required(true)
        })
        .create_option(|option| {
            option
                .name(constant::value::NEGATIVE_PROMPT)
                .description("The prompt to avoid drawing")
                .kind(CommandOptionType::String)
                .required(false)
        })
        .create_option(|option| {
            option
                .name(constant::value::SEED)
                .description("The seed to use")
                .kind(CommandOptionType::Integer)
                .required(false)
        })
        .create_option(|option| {
            option
                .name(constant::value::COUNT)
                .description("The number of images to generate")
                .kind(CommandOptionType::Integer)
                .min_int_value(constant::limits::COUNT_MIN)
                .max_int_value(constant::limits::COUNT_MAX)
                .required(false)
        })
        .create_option(|option| {
            option
                .name(constant::value::WIDTH)
                .description("The width of the image")
                .kind(CommandOptionType::Integer)
                .min_int_value(constant::limits::WIDTH_MIN)
                .max_int_value(constant::limits::WIDTH_MAX)
                .required(false)
        })
        .create_option(|option| {
            option
                .name(constant::value::HEIGHT)
                .description("The height of the image")
                .kind(CommandOptionType::Integer)
                .min_int_value(constant::limits::HEIGHT_MIN)
                .max_int_value(constant::limits::HEIGHT_MAX)
                .required(false)
        })
        .create_option(|option| {
            option
                .name(constant::value::GUIDANCE_SCALE)
                .description("The scale of the guidance to apply")
                .kind(CommandOptionType::Number)
                .min_number_value(constant::limits::GUIDANCE_SCALE_MIN)
                .max_number_value(constant::limits::GUIDANCE_SCALE_MAX)
                .required(false)
        })
        .create_option(|option| {
            option
                .name(constant::value::STEPS)
                .description("The number of denoising steps to apply")
                .kind(CommandOptionType::Integer)
                .min_int_value(constant::limits::STEPS_MIN)
                .max_int_value(constant::limits::STEPS_MAX)
                .required(false)
        })
        .create_option(|option| {
            option
                .name(constant::value::TILING)
                .description("Whether or not the image should be tiled at the edges")
                .kind(CommandOptionType::Boolean)
                .required(false)
        })
        .create_option(|option| {
            option
                .name(constant::value::RESTORE_FACES)
                .description("Whether or not the image should have its faces restored")
                .kind(CommandOptionType::Boolean)
                .required(false)
        })
        .create_option(|option| {
            option
                .name(constant::value::DENOISING_STRENGTH)
                .description(
                    "The amount of denoising to apply (0 is no change, 1 is complete remake)",
                )
                .kind(CommandOptionType::Number)
                .min_number_value(0.0)
                .max_number_value(1.0)
                .required(false)
        })
        .create_option(|option| {
            let opt = option
                .name(constant::value::SAMPLER)
                .description("The sampler to use")
                .kind(CommandOptionType::String)
                .required(false);

            for value in sd::Sampler::VALUES {
                opt.add_string_choice(value, value);
            }

            opt
        });

    for (idx, chunk) in models.chunks(constant::misc::MODEL_CHUNK_COUNT).enumerate() {
        command.create_option(|option| {
            let opt = option
                .name(if idx == 0 {
                    constant::value::MODEL.to_string()
                } else {
                    format!("{}{}", constant::value::MODEL, idx + 1)
                })
                .description(format!("The model to use, category {}", idx + 1))
                .kind(CommandOptionType::String)
                .required(false);

            for model in chunk {
                opt.add_string_choice(&model.name, &model.title);
            }

            opt
        });
    }
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
}
impl<'a> Overrides<'a> {
    fn new(
        prompt: Option<&'a str>,
        negative_prompt: Option<&'a str>,
        width: Option<u32>,
        height: Option<u32>,
        guidance_scale: Option<f64>,
        steps: Option<usize>,
        seed: Option<Option<i64>>,
        denoising_strength: Option<f64>,
    ) -> Self {
        use constant::limits as L;

        let (width, height) = if let (Some(width), Some(height)) = (width, height) {
            let (width, height) = util::fixup_resolution(width, height);
            (Some(width), Some(height))
        } else {
            (None, None)
        };

        Self {
            prompt: prompt.filter(|s| !s.is_empty()),
            negative_prompt: negative_prompt.filter(|s| !s.is_empty()),
            width,
            height,
            guidance_scale: guidance_scale
                .map(|s| s.clamp(L::GUIDANCE_SCALE_MIN, L::GUIDANCE_SCALE_MAX)),
            steps: steps.map(|s| s.clamp(L::STEPS_MIN, L::STEPS_MAX)),
            seed,
            denoising_strength: denoising_strength.map(|s| s.clamp(0.0, 1.0)),
        }
    }

    fn none() -> Self {
        Self {
            prompt: None,
            negative_prompt: None,
            width: None,
            height: None,
            guidance_scale: None,
            steps: None,
            seed: Some(None),
            denoising_strength: None,
        }
    }
}
