use crate::{command, config::Configuration, constant, store, util};

use super::Session;
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

                command::populate_generate_options(
                    |opt| {
                        o.add_sub_option(opt);
                    },
                    models,
                    false,
                );

                o.create_sub_option(|o| {
                    o.kind(CommandOptionType::String)
                        .name(constant::value::TAGS_URL)
                        .description(
                            "The URL of tags to use (defaults to Danbooru safe if not specified)",
                        )
                })
                .create_sub_option(|o| {
                    o.kind(CommandOptionType::Boolean)
                        .name(constant::value::HIDE_PROMPT)
                        .description("Whether or not to hide the prompt for generations")
                });

                o
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
) -> anyhow::Result<()> {
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
) -> anyhow::Result<()> {
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

    let url = util::get_value(&subcommand.options, constant::value::TAGS_URL)
        .and_then(util::value_to_string);

    let hide_prompt = util::get_value(&subcommand.options, constant::value::HIDE_PROMPT)
        .and_then(util::value_to_bool)
        .unwrap_or(false);

    let params = command::OwnedBaseGenerationParameters::load(
        cmd.user.id,
        &subcommand.options,
        store,
        models,
        false,
        false,
    )?;

    fn display<T: Display>(value: &Option<T>) -> Option<&dyn Display> {
        value.as_ref().map(|s| s as &dyn Display)
    }

    cmd.create_interaction_response(&http, |response| {
        response
            .kind(InteractionResponseType::ChannelMessageWithSource)
            .interaction_response_data(|message| {
                message.content(format!(
                    "Starting with the following settings:\n{}",
                    [
                        (
                            "Tags",
                            if let Some(url) = url.as_ref() {
                                Some(url as &dyn Display)
                            } else {
                                Some(&"Danbooru tags" as &dyn Display)
                            },
                        ),
                        ("Negative prompt", display(&params.negative_prompt)),
                        ("Seed", display(&params.seed)),
                        ("Count", display(&params.batch_count)),
                        ("Width", display(&params.width)),
                        ("Height", display(&params.height)),
                        ("Guidance scale", display(&params.cfg_scale)),
                        ("Denoising strength", display(&params.denoising_strength)),
                        ("Steps", display(&params.steps)),
                        ("Tiling", display(&params.tiling)),
                        ("Restore faces", display(&params.restore_faces)),
                        ("Sampler", display(&params.sampler)),
                        (
                            "Model",
                            display(&params.model.as_ref().map(|s| &s.name as &dyn Display))
                        ),
                    ]
                    .into_iter()
                    .filter_map(|(key, value)| Some((key, value?)))
                    .map(|(key, value)| format!("- *{key}*: {value}"))
                    .collect::<Vec<_>>()
                    .join("\n")
                ))
            })
    })
    .await?;

    let tags = if let Some(url) = url {
        reqwest::get(url)
            .await?
            .text()
            .await?
            .lines()
            .map(|l| l.trim().to_string())
            .collect()
    } else {
        Configuration::get()
            .deepdanbooru_tag_whitelist()
            .map(|w| w.iter().cloned().collect())
            .unwrap_or_default()
    };

    sessions.lock().insert(
        cmd.channel_id,
        super::Session::new(
            http,
            cmd.channel_id,
            client.clone(),
            params,
            tags,
            hide_prompt,
        )?,
    );
    Ok(())
}

async fn stop(
    http: &Http,
    cmd: &ApplicationCommandInteraction,
    sessions: &Mutex<HashMap<ChannelId, Session>>,
) -> anyhow::Result<()> {
    let session = sessions.lock().remove(&cmd.channel_id);
    let Some(session) = session else {
            cmd.create_interaction_response(http, |response| {
                response
                    .kind(InteractionResponseType::ChannelMessageWithSource)
                    .interaction_response_data(|message| {
                        message.content("No session running...")
                    })
            })
            .await?;
            return Ok(());
        };

    session.shutdown();
    std::mem::drop(session);

    cmd.create_interaction_response(http, |response| {
        response
            .kind(InteractionResponseType::ChannelMessageWithSource)
            .interaction_response_data(|message| {
                message.content("Session terminated. You are now free to start again.")
            })
    })
    .await?;

    Ok(())
}
