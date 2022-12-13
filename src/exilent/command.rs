use super::issuer;
use crate::{
    command, constant, custom_id as cid, store,
    util::{self, DiscordInteraction},
};
use anyhow::Context;
use serenity::{
    http::Http,
    model::prelude::{
        command::{Command, CommandOptionType},
        interaction::{
            application_command::ApplicationCommandInteraction, InteractionResponseType,
        },
        *,
    },
};
use stable_diffusion_a1111_webui_client as sd;
use std::collections::HashMap;

pub async fn register(http: &Http, models: &[sd::Model]) -> anyhow::Result<()> {
    Command::create_global_application_command(http, |command| {
        command
            .name(constant::command::PAINT)
            .description("Paints your dreams");

        command::populate_generate_options(
            |opt| {
                command.add_option(opt);
            },
            models,
            true,
        );
        command
    })
    .await?;

    Command::create_global_application_command(http, |command| {
        command
            .name(constant::command::PAINTOVER)
            .description("Paints your dreams over another image");

        command::populate_generate_options(
            |opt| {
                command.add_option(opt);
            },
            models,
            true,
        );
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
    .await?;

    Command::create_global_application_command(http, |command| {
        command
            .name(constant::command::PAINTAGAIN)
            .description("Re-run the last generation command with a modal of overrides")
    })
    .await?;

    Command::create_global_application_command(http, |command| {
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
    .await?;

    Command::create_global_application_command(http, |command| {
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
    .await?;

    Command::create_global_application_command(http, |command| {
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
    .await?;

    Command::create_global_application_command(http, |command| {
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
    .await?;

    Ok(())
}

pub async fn exilent(
    client: &sd::Client,
    models: &[sd::Model],
    store: &store::Store,
    http: &Http,
    cmd: ApplicationCommandInteraction,
) -> anyhow::Result<()> {
    match cmd.data.options[0].name.as_str() {
        "embeddings" => embeddings(client, http, cmd).await,
        "stats" => stats(models, store, http, cmd).await,
        _ => unreachable!(),
    }
}

async fn embeddings(
    client: &sd::Client,
    http: &Http,
    cmd: ApplicationCommandInteraction,
) -> anyhow::Result<()> {
    cmd.create_interaction_response(http, |response| {
        response
            .kind(InteractionResponseType::ChannelMessageWithSource)
            .interaction_response_data(|message| {
                message.title("Embeddings").content("Processing...")
            })
    })
    .await?;
    let texts = match client.embeddings().await {
        Ok(embeddings) => {
            util::generate_chunked_strings(embeddings.iter().map(|s| format!("`{s}`")), 1900)
        }
        Err(err) => vec![format!("{err:?}")],
    };
    cmd.edit(http, texts.first().map(|s| s.as_str()).unwrap_or_default())
        .await?;

    for remainder in texts.iter().skip(1) {
        cmd.channel_id
            .send_message(http, |msg| msg.content(remainder))
            .await?;
    }

    Ok(())
}

async fn stats(
    models: &[sd::Model],
    store: &store::Store,
    http: &Http,
    cmd: ApplicationCommandInteraction,
) -> anyhow::Result<()> {
    let stats = store.get_model_usage_counts()?;
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
            .chain(counts.into_iter().map(|(model_hash, count)| {
                format!(
                    "- {}: {} generations",
                    util::find_model_by_hash(models, &model_hash)
                        .as_ref()
                        .map(|m| m.1.name.as_str())
                        .unwrap_or("unknown model"),
                    count
                )
            }))
        })
        .collect::<Vec<String>>()
        .join("\n");

    cmd.create(http, &message).await?;
    Ok(())
}

pub async fn paint(
    client: &sd::Client,
    models: &[sd::Model],
    store: &store::Store,
    http: &Http,
    aci: ApplicationCommandInteraction,
) -> anyhow::Result<()> {
    let interaction: &dyn DiscordInteraction = &aci;
    interaction
        .create(http, "Paint request received, processing...")
        .await?;
    let mut base_parameters = command::OwnedBaseGenerationParameters::load(
        aci.user.id,
        &aci.data.options,
        store,
        models,
        true,
        true,
    )?;

    if let Some(model) = &base_parameters.model {
        base_parameters.prompt =
            util::prepend_keyword_if_necessary(&base_parameters.prompt, &model.name);
    };

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
        client.generate_from_text(&sd::TextToImageGenerationRequest {
            base: base_parameters.as_base_generation_request(),
            ..Default::default()
        })?,
        models,
        store,
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

pub async fn paintover(
    client: &sd::Client,
    models: &[sd::Model],
    store: &store::Store,
    http: &Http,
    aci: ApplicationCommandInteraction,
) -> anyhow::Result<()> {
    let interaction: &dyn DiscordInteraction = &aci;
    interaction
        .create(http, "Paintover request received, processing...")
        .await?;
    let mut base_parameters = command::OwnedBaseGenerationParameters::load(
        aci.user.id,
        &aci.data.options,
        store,
        models,
        false,
        true,
    )?;
    let url = util::get_image_url(&aci)?;
    let resize_mode = util::get_value(&aci.data.options, constant::value::RESIZE_MODE)
        .and_then(util::value_to_string)
        .and_then(|s| sd::ResizeMode::try_from(s.as_str()).ok())
        .unwrap_or_default();

    if let Some(model) = &base_parameters.model {
        base_parameters.prompt =
            util::prepend_keyword_if_necessary(&base_parameters.prompt, &model.name);
    };

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
        client.generate_from_image_and_text(&sd::ImageToImageGenerationRequest {
            base: base_parameters.as_base_generation_request(),
            images: vec![image.clone()],
            resize_mode: Some(resize_mode),
            ..Default::default()
        })?,
        models,
        store,
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

pub async fn paintagain(
    store: &store::Store,
    http: &Http,
    aci: ApplicationCommandInteraction,
) -> anyhow::Result<()> {
    let generation = store
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

pub async fn postprocess(
    client: &sd::Client,
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

    let options = &aci.data.options;

    let upscaler_1 = util::get_value(options, constant::value::UPSCALER_1)
        .and_then(util::value_to_string)
        .and_then(|v| sd::Upscaler::try_from(v.as_str()).ok())
        .context("expected upscaler 1")?;

    let upscaler_2 = util::get_value(options, constant::value::UPSCALER_2)
        .and_then(util::value_to_string)
        .and_then(|v| sd::Upscaler::try_from(v.as_str()).ok())
        .context("expected upscaler 2")?;

    let scale_factor = util::get_value(options, constant::value::SCALE_FACTOR)
        .and_then(util::value_to_number)
        .map(|n| n as f32)
        .context("expected scale factor")?;

    let codeformer_visibility = util::get_value(options, constant::value::CODEFORMER_VISIBILITY)
        .and_then(util::value_to_number)
        .map(|n| n as f32);

    let codeformer_weight = util::get_value(options, constant::value::CODEFORMER_WEIGHT)
        .and_then(util::value_to_number)
        .map(|n| n as f32);

    let upscaler_2_visibility = util::get_value(options, constant::value::UPSCALER_2_VISIBILITY)
        .and_then(util::value_to_number)
        .map(|n| n as f32);

    let gfpgan_visibility = util::get_value(options, constant::value::GFPGAN_VISIBILITY)
        .and_then(util::value_to_number)
        .map(|n| n as f32);

    let upscale_first =
        util::get_value(options, constant::value::UPSCALE_FIRST).and_then(util::value_to_bool);

    let result = client
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

pub async fn interrogate(
    client: &sd::Client,
    store: &store::Store,
    http: &Http,
    aci: ApplicationCommandInteraction,
) -> anyhow::Result<()> {
    let url = util::get_image_url(&aci)?;

    let interrogator = util::get_value(&aci.data.options, constant::value::INTERROGATOR)
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
        client,
        store,
        interaction,
        http,
        (image, store::InterrogationSource::Url(url), interrogator),
    )
    .await
}

pub async fn png_info(
    client: &sd::Client,
    http: &Http,
    aci: ApplicationCommandInteraction,
) -> anyhow::Result<()> {
    let url = util::get_image_url(&aci)?;

    let interaction: &dyn DiscordInteraction = &aci;
    interaction
        .create(http, &format!("Reading PNG info of {url}..."))
        .await?;

    let bytes = reqwest::get(&url).await?.bytes().await?;
    let result = client.png_info(&bytes).await?;
    interaction.edit(http, &result).await?;

    Ok(())
}