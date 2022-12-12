use crate::constant;

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
};
use stable_diffusion_a1111_webui_client as sd;
use std::{collections::HashMap, sync::Arc};

pub const MODEL_NAME_PREFIX: &str = "Anything";

pub async fn register(http: &Http) -> anyhow::Result<()> {
    Command::create_global_application_command(http, |command| {
        command
            .name(constant::command::WIREHEAD)
            .description("Interact with Wirehead")
            .create_option(|o| {
                o.kind(CommandOptionType::SubCommand)
                    .name("start")
                    .description("Start a Wirehead session (if not already running)")
                    .create_sub_option(|o| {
                        o.kind(CommandOptionType::String).name("url").description(
                            "The URL of tags to use (defaults to Danbooru safe if not specified)",
                        )
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
    sessions: &Mutex<HashMap<ChannelId, Session>>,
    client: Arc<sd::Client>,
    models: &[sd::Model],
    http: Arc<Http>,
    cmd: ApplicationCommandInteraction,
) -> anyhow::Result<()> {
    let subcommand = &cmd.data.options[0];
    match subcommand.name.as_str() {
        "start" => start(http, &cmd, subcommand, sessions, client, models).await,
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

    let url = subcommand
        .options
        .iter()
        .find(|o| o.name == "url")
        .and_then(|o| o.value.as_ref())
        .and_then(|v| v.as_str());

    cmd.create_interaction_response(&http, |response| {
        response
            .kind(InteractionResponseType::ChannelMessageWithSource)
            .interaction_response_data(|message| {
                message.content(format!(
                    "Starting with {}...",
                    if let Some(url) = url {
                        url
                    } else {
                        "Danbooru tags"
                    }
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
        constant::resource::DANBOORU_TAGS
            .iter()
            .map(|s| s.to_string())
            .collect()
    };

    let model = models
        .iter()
        .find(|m| m.name.starts_with(MODEL_NAME_PREFIX))
        .context("failed to find model")?;

    sessions.lock().insert(
        cmd.channel_id,
        super::Session::new(
            http,
            cmd.channel_id,
            client.clone(),
            Arc::new(model.clone()),
            tags,
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
