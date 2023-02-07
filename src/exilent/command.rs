use super::issuer;
use crate::{
    command,
    config::Configuration,
    constant, store,
    util::{self, DiscordInteraction},
};
use anyhow::Context;
use itertools::Itertools;
use serenity::{
    http::Http,
    model::prelude::{
        command::{Command, CommandOptionType},
        interaction::application_command::ApplicationCommandInteraction,
        *,
    },
};
use stable_diffusion_a1111_webui_client as sd;

pub async fn register(http: &Http, models: &[sd::Model]) -> anyhow::Result<()> {
    Command::create_global_application_command(http, |command| {
        command
            .name(&Configuration::get().commands.paint)
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
            .name(&Configuration::get().commands.postprocess)
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
            .name(&Configuration::get().commands.interrogate)
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
            .name(&Configuration::get().commands.exilent)
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
            .name(&Configuration::get().commands.png_info)
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
) {
    match cmd.data.options[0].name.as_str() {
        "embeddings" => embeddings(client, http, cmd).await,
        "stats" => stats(models, store, http, cmd).await,
        _ => unreachable!(),
    }
}

async fn embeddings(client: &sd::Client, http: &Http, cmd: ApplicationCommandInteraction) {
    cmd.create(http, "Getting embeddings...").await.unwrap();

    util::run_and_report_error(&cmd, http, async {
        let embeddings = match client.embeddings().await {
            Ok(embeddings) => embeddings
                .all()
                .map(|(s, _)| format!("`{s}`"))
                .sorted()
                .collect(),
            Err(err) => vec![format!("{err:?}")],
        };
        util::chunked_response(http, &cmd, embeddings.iter().map(|s| s.as_str()), ", ").await?;

        Ok(())
    })
    .await;
}

async fn stats(
    models: &[sd::Model],
    store: &store::Store,
    http: &Http,
    cmd: ApplicationCommandInteraction,
) {
    cmd.create(http, "Getting stats...").await.unwrap();

    util::run_and_report_error(&cmd, http, async {
        let stats = store.get_model_usage_counts(cmd.guild_id.context("no guild id")?)?;
        async fn get_user_name(
            http: &Http,
            guild_id: Option<GuildId>,
            id: UserId,
        ) -> anyhow::Result<(String, UserId)> {
            let user = id.to_user(http).await?;
            let name = if let Some(guild_id) = guild_id {
                user.nick_in(http, guild_id).await
            } else {
                None
            };
            Ok((name.unwrap_or(user.name), id))
        }

        let mut users = futures::future::join_all(
            stats
                .keys()
                .map(|id| get_user_name(http, cmd.guild_id, *id)),
        )
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
        users.sort_by(|a, b| a.0.cmp(&b.0));

        let guild = http
            .get_guild(*cmd.guild_id.context("no guild id")?.as_u64())
            .await?;
        let header = format!("**Statistics for server *{}***:", guild.name);

        let mut body = users
            .into_iter()
            .flat_map(|(user_name, user_id)| {
                std::iter::once(format!("**{user_name}**"))
                    .chain(
                        stats
                            .get(&user_id)
                            .unwrap()
                            .iter()
                            .map(|(model_hash, count)| {
                                format!(
                                    "- {}: {} generations",
                                    util::find_model_by_hash(models, model_hash)
                                        .as_ref()
                                        .map(|m| m.1.name.clone())
                                        .unwrap_or_else(|| format!("unknown model [{model_hash}]")),
                                    count
                                )
                            }),
                    )
                    .chain(std::iter::once(String::new()))
            })
            .collect::<Vec<String>>();

        let mut message = vec![header];
        if body.is_empty() {
            message.push("No generations yet.".to_string());
        } else {
            message.append(&mut body);
        }

        util::chunked_response(http, &cmd, message.iter().map(|s| s.as_str()), "\n").await
    })
    .await;
}

pub async fn paint(
    client: &sd::Client,
    models: &[sd::Model],
    store: &store::Store,
    http: &Http,
    aci: ApplicationCommandInteraction,
) {
    aci.create(http, "Paint request received, processing...")
        .await
        .unwrap();

    util::run_and_report_error(&aci, http, async {
        let params = command::GenerationParameters::load(
            aci.user().id,
            aci.guild_id().context("no guild id")?,
            &aci.data.options,
            store,
            models,
            true,
            true,
        )
        .await?;

        let base = params.base_generation();
        aci.edit(
            http,
            &format!(
                "`{}`{}: Generating (waiting for start)...",
                &base.prompt,
                base.negative_prompt
                    .as_ref()
                    .filter(|s| !s.is_empty())
                    .map(|s| format!(" - `{s}`"))
                    .unwrap_or_default()
            ),
        )
        .await?;

        let (prompt, negative_prompt) = (base.prompt.clone(), base.negative_prompt.clone());
        issuer::generation_task(
            (client, models),
            tokio::task::spawn(params.clone().generate(client)),
            store,
            http,
            (&aci, None),
            (&prompt, negative_prompt.as_deref()),
            params.image_generation(),
        )
        .await
    })
    .await;
}

pub async fn postprocess(client: &sd::Client, http: &Http, aci: ApplicationCommandInteraction) {
    aci.create(http, "Postprocess request received, processing...")
        .await
        .unwrap();

    util::run_and_report_error(&aci, http, async {
        let options = &aci.data.options;
        let url = util::get_image_url(options).context("no url specified")?;

        aci.edit(http, &format!("Postprocessing {url}...")).await?;

        let bytes = reqwest::get(&url).await?.bytes().await?;
        let image = image::load_from_memory(&bytes)?;

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

        let codeformer_visibility =
            util::get_value(options, constant::value::CODEFORMER_VISIBILITY)
                .and_then(util::value_to_number)
                .map(|n| n as f32);

        let codeformer_weight = util::get_value(options, constant::value::CODEFORMER_WEIGHT)
            .and_then(util::value_to_number)
            .map(|n| n as f32);

        let upscaler_2_visibility =
            util::get_value(options, constant::value::UPSCALER_2_VISIBILITY)
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

        aci.get_interaction_message(http)
            .await?
            .edit(http, |m| {
                m.content(format!("Postprocessing of <{url}> complete."))
                    .attachment((bytes.as_slice(), "postprocess.png"))
            })
            .await?;

        Ok(())
    })
    .await;
}

pub async fn interrogate(
    client: &sd::Client,
    store: &store::Store,
    http: &Http,
    aci: ApplicationCommandInteraction,
) {
    aci.create(http, "Interrogation request received, processing...")
        .await
        .unwrap();

    util::run_and_report_error(&aci, http, async {
        let url = util::get_image_url(&aci.data.options).context("no url specified")?;

        let interrogator = util::get_value(&aci.data.options, constant::value::INTERROGATOR)
            .and_then(util::value_to_string)
            .and_then(|v| sd::Interrogator::try_from(v.as_str()).ok())
            .context("expected interrogator")?;

        aci.edit(http, &format!("Interrogating {url} with {interrogator}..."))
            .await?;

        let bytes = reqwest::get(&url).await?.bytes().await?;
        let image = image::load_from_memory(&bytes)?;

        issuer::interrogate_task(
            client,
            store,
            &aci,
            http,
            (image, store::InterrogationSource::Url(url), interrogator),
        )
        .await?;

        Ok(())
    })
    .await;
}

pub async fn png_info(client: &sd::Client, http: &Http, aci: ApplicationCommandInteraction) {
    aci.create(http, "PNG info request received, processing...")
        .await
        .unwrap();

    util::run_and_report_error(&aci, http, async {
        let url = util::get_image_url(&aci.data.options).context("no url specified")?;

        let interaction: &dyn DiscordInteraction = &aci;
        interaction
            .edit(http, &format!("Reading PNG info of {url}..."))
            .await?;

        let bytes = reqwest::get(&url).await?.bytes().await?;
        let result = client.png_info(&bytes).await?;
        interaction.edit(http, &result).await?;

        Ok(())
    })
    .await;
}
