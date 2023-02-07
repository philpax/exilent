use crate::{
    command,
    config::Configuration,
    constant, store,
    util::{self, DiscordInteraction},
};

use super::Session;
use anyhow::Context;
use parking_lot::Mutex;
use serenity::{
    http::Http,
    model::prelude::{
        command::{Command, CommandOptionType},
        interaction::{
            application_command::{ApplicationCommandInteraction, CommandDataOption},
            InteractionResponseType,
        },
        ChannelId,
    },
    prelude::Mentionable,
};
use stable_diffusion_a1111_webui_client as sd;
use std::{collections::HashMap, fmt::Display, sync::Arc};

pub async fn register(http: &Http, models: &[sd::Model]) -> anyhow::Result<()> {
    Command::create_global_application_command(http, |command| {
        command
            .name(&Configuration::get().commands.wirehead)
            .description("Interact with Wirehead")
            .create_option(|o| {
                o.kind(CommandOptionType::SubCommand)
                    .name("start")
                    .description("Start a Wirehead session (if not already running)");

                o.create_sub_option(|o| {
                    o.kind(CommandOptionType::String)
                        .name(constant::value::TAGS)
                        .description("The tags to use for generation")
                        .required(true);

                    for tag_list_name in Configuration::get().tags().keys() {
                        o.add_string_choice(tag_list_name, tag_list_name);
                    }

                    o
                });

                o.create_sub_option(|o| {
                    o.kind(CommandOptionType::Boolean)
                        .name(constant::value::TO_EXILENT_ENABLED)
                        .description("Whether or not the To Exilent button post-rating is shown")
                        .required(true)
                });

                command::populate_generate_options(
                    |opt| {
                        o.add_sub_option(opt);
                    },
                    models,
                    false,
                );

                o.create_sub_option(|o| {
                    o.kind(CommandOptionType::Boolean)
                        .name(constant::value::HIDE_PROMPT)
                        .description("Whether or not to hide the prompt for generations")
                }).create_sub_option(|o| {
                    o.kind(CommandOptionType::Channel)
                        .name(constant::value::TO_EXILENT_CHANNEL)
                        .description("The channel to send To Exilent results to. If not set, results will be sent to the same channel.")
                }).create_sub_option(|o| {
                    o.kind(CommandOptionType::String)
                        .name(constant::value::PREFIX)
                        .description("A prefix to add to the generation prompt. (Will be joined by a comma)")
                }).create_sub_option(|o| {
                    o.kind(CommandOptionType::String)
                        .name(constant::value::SUFFIX)
                        .description("A prefix to add to the generation prompt. (Will be joined by a comma)")
                })
            })
            .create_option(|o| {
                o.kind(CommandOptionType::SubCommand)
                    .name("stop")
                    .description("Stop a Wirehead session (if running)")
            })
    })
    .await?;

    Ok(())
}

pub async fn wirehead(
    http: Arc<Http>,
    cmd: ApplicationCommandInteraction,
    sessions: &Mutex<HashMap<ChannelId, Session>>,
    client: Arc<sd::Client>,
    models: &[sd::Model],
    store: &store::Store,
) {
    let subcommand = &cmd.data.options[0];
    match subcommand.name.as_str() {
        "start" => start(http, &cmd, subcommand, sessions, client, models, store).await,
        "stop" => stop(&http, &cmd, sessions).await,
        _ => unreachable!(),
    }
}

async fn start(
    http: Arc<Http>,
    cmd: &ApplicationCommandInteraction,
    subcommand: &CommandDataOption,
    sessions: &Mutex<HashMap<ChannelId, Session>>,
    client: Arc<sd::Client>,
    models: &[sd::Model],
    store: &store::Store,
) {
    cmd.create(&http, "Starting...").await.unwrap();

    util::run_and_report_error(cmd, http.clone().as_ref(), async {
        if sessions.lock().contains_key(&cmd.channel_id) {
            cmd.create_interaction_response(&http, |response| {
                response
                    .kind(InteractionResponseType::ChannelMessageWithSource)
                    .interaction_response_data(|message| {
                        message.content("Session already under way...")
                    })
            })
            .await?;
            return Ok(());
        }

        let tag_selection = util::get_value(&subcommand.options, constant::value::TAGS)
            .and_then(util::value_to_string)
            .context("no tag selection")?;

        let hide_prompt = util::get_value(&subcommand.options, constant::value::HIDE_PROMPT)
            .and_then(util::value_to_bool)
            .unwrap_or(false);

        let to_exilent_enabled =
            util::get_value(&subcommand.options, constant::value::TO_EXILENT_ENABLED)
                .and_then(util::value_to_bool)
                .unwrap();

        let to_exilent_channel =
            util::get_value(&subcommand.options, constant::value::TO_EXILENT_CHANNEL)
                .and_then(util::value_to_channel)
                .map(|c| c.id);
        if !to_exilent_enabled && to_exilent_channel.is_some() {
            anyhow::bail!("a To Exilent channel was set, but To Exilent is not enabled");
        }
        let to_exilent_channel_id = to_exilent_channel
            .or(Some(cmd.channel_id))
            .filter(|_| to_exilent_enabled);

        let prefix = util::get_value(&subcommand.options, constant::value::PREFIX)
            .and_then(util::value_to_string);
        let suffix = util::get_value(&subcommand.options, constant::value::SUFFIX)
            .and_then(util::value_to_string);

        let parameters = command::GenerationParameters::load(
            cmd.user.id,
            cmd.guild_id.context("no guild id")?,
            &subcommand.options,
            store,
            models,
            false,
            false,
        )
        .await?;

        fn display<T: Display>(value: &Option<T>) -> Option<&dyn Display> {
            value.as_ref().map(|s| s as &dyn Display)
        }

        let base = parameters.base_generation();
        let (image_url, resize_mode) = parameters.image_params().unzip();
        cmd.edit(
            &http,
            &format!(
                "Starting with the following settings:\n{}",
                [
                    ("Tags", Some(&tag_selection as &dyn Display)),
                    ("Prefix", display(&prefix)),
                    ("Suffix", display(&suffix)),
                    ("Image URL", display(&image_url)),
                    ("Negative prompt", display(&base.negative_prompt)),
                    ("Seed", display(&base.seed)),
                    ("Count", display(&base.batch_count)),
                    ("Width", display(&base.width)),
                    ("Height", display(&base.height)),
                    ("Guidance scale", display(&base.cfg_scale)),
                    ("Denoising strength", display(&base.denoising_strength)),
                    ("Resize mode", display(&resize_mode)),
                    ("Steps", display(&base.steps)),
                    ("Tiling", display(&base.tiling)),
                    ("Restore faces", display(&base.restore_faces)),
                    ("Sampler", display(&base.sampler)),
                    (
                        "Model",
                        display(&base.model.as_ref().map(|m| m.name.as_str()))
                    ),
                    (
                        "To Exilent channel",
                        display(&to_exilent_channel_id.map(|c| c.mention()))
                    )
                ]
                .into_iter()
                .filter_map(|(key, value)| Some((key, value?)))
                .map(|(key, value)| format!("- *{key}*: {value}"))
                .collect::<Vec<_>>()
                .join("\n")
            ),
        )
        .await?;

        let tags = Configuration::get()
            .tags()
            .get(&tag_selection)
            .context("invalid tag selection")?
            .iter()
            .cloned()
            .collect();

        let original_message_link = cmd.get_interaction_response(&http).await?.link();
        sessions.lock().insert(
            cmd.channel_id,
            super::Session::new(
                http,
                cmd.channel_id,
                to_exilent_channel_id,
                client.clone(),
                hide_prompt,
                super::GenerationParameters {
                    parameters,
                    tags,
                    prefix,
                    suffix,
                },
                original_message_link,
            )?,
        );
        Ok(())
    })
    .await;
}

async fn stop(
    http: &Http,
    cmd: &ApplicationCommandInteraction,
    sessions: &Mutex<HashMap<ChannelId, Session>>,
) {
    cmd.create(http, "Attemping to stop Wirehead session...")
        .await
        .unwrap();

    util::run_and_report_error(cmd, http, async {
        let session = match sessions.lock().remove(&cmd.channel_id) {
            Some(session) => session,
            _ => {
                anyhow::bail!("No Wirehead session running!");
            }
        };

        session.shutdown();
        cmd.edit(
            http,
            &format!(
                "Wirehead session ({}) terminated. You are now free to start again.",
                session.original_message_link
            ),
        )
        .await?;
        std::mem::drop(session);

        Ok(())
    })
    .await;
}
