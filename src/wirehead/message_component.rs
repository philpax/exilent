use super::{simulation::AsPhenotype, GenerationParameters, Session, TextGenome};
use crate::{
    custom_id as cid, exilent, store,
    util::{self, DiscordInteraction},
};
use parking_lot::Mutex;
use serenity::{
    http::Http,
    model::prelude::{
        component::ButtonStyle,
        interaction::{message_component::MessageComponentInteraction, InteractionResponseType},
        ChannelId,
    },
    prelude::Mentionable,
};
use stable_diffusion_a1111_webui_client as sd;
use std::collections::HashMap;

pub async fn to_exilent(
    sessions: &Mutex<HashMap<ChannelId, Session>>,
    store: &store::Store,
    (client, models): (&sd::Client, &[sd::Model]),
    http: &Http,
    mci: MessageComponentInteraction,
    genome: TextGenome,
    seed: i64,
) {
    mci.create(http, "Generating Wirehead image with Exilent...")
        .await
        .unwrap();
    util::run_and_report_error(&mci, http, async {
        if !sessions.lock().contains_key(&mci.channel_id) {
            anyhow::bail!("There is no active Wirehead session.");
        };

        let (to_exilent_channel_id, parameters) = sessions
            .lock()
            .get(&mci.channel_id)
            .map(|s| (s.to_exilent_channel_id, s.generation_parameters.clone()))
            .unwrap();

        mci.edit(
            http,
            &format!(
                "Generating with Exilent{} (waiting for start)...",
                to_exilent_channel_id
                    .map(|c| format!(" in {}", c.mention()))
                    .unwrap_or_default()
            ),
        )
        .await?;

        let GenerationParameters {
            parameters,
            tags,
            prefix,
            suffix,
        } = parameters;

        let base = {
            let mut base = sd::BaseGenerationRequest {
                prompt: genome.as_text(&tags, prefix.as_deref(), suffix.as_deref()),
                ..parameters.as_base_generation_request()
            };
            util::fixup_base_generation_request(&mut base);
            base.seed = Some(seed);
            base
        };

        let (prompt, negative_prompt) = (base.prompt.clone(), base.negative_prompt.clone());
        exilent::issuer::generation_task(
            (client, models),
            tokio::task::spawn(
                client.generate_from_text(&sd::TextToImageGenerationRequest {
                    base,
                    ..Default::default()
                }),
            ),
            store,
            http,
            (&mci, to_exilent_channel_id),
            (&prompt, negative_prompt.as_deref()),
            None,
        )
        .await?;

        Ok(())
    })
    .await;
}

pub async fn rate(
    sessions: &Mutex<HashMap<ChannelId, Session>>,
    http: &Http,
    mci: MessageComponentInteraction,
    genome: TextGenome,
    custom_id: cid::Wirehead,
) {
    util::run_and_report_error(&mci, http, async {
        if !sessions.lock().contains_key(&mci.channel_id) {
            anyhow::bail!("There is no active Wirehead session.");
        };

        // this is a bit of a contortion but it's fine for now
        let (hide_prompt, to_exilent_enabled, params) = sessions
            .lock()
            .get(&mci.channel_id)
            .map(|session| {
                session.rate(
                    genome.clone(),
                    match custom_id.value {
                        cid::WireheadValue::Negative2 => 0,
                        cid::WireheadValue::Negative1 => 25,
                        cid::WireheadValue::Zero => 50,
                        cid::WireheadValue::Positive1 => 75,
                        cid::WireheadValue::Positive2 => 100,
                        cid::WireheadValue::ToExilent => unreachable!(),
                    },
                );
                (
                    session.hide_prompt,
                    session.to_exilent_channel_id.is_some(),
                    session.generation_parameters.clone(),
                )
            })
            .unwrap();

        mci.create_interaction_response(http, |m| {
            m.kind(InteractionResponseType::UpdateMessage)
                .interaction_response_data(|d| {
                    d.content(format!(
                        "{}**Rating**: {} by {}",
                        if !hide_prompt {
                            format!(
                                "`{}` | ",
                                genome.as_text(
                                    &params.tags,
                                    params.prefix.as_deref(),
                                    params.suffix.as_deref()
                                )
                            )
                        } else {
                            String::new()
                        },
                        custom_id.value.as_integer(),
                        mci.user.mention(),
                    ))
                    .components(|c| {
                        if to_exilent_enabled {
                            c.create_action_row(|row| {
                                row.create_button(|b| {
                                    b.custom_id(
                                        cid::WireheadValue::ToExilent
                                            .to_id(genome.clone(), custom_id.seed),
                                    )
                                    .label("To Exilent")
                                    .style(ButtonStyle::Primary)
                                })
                            })
                        } else {
                            c
                        }
                    })
                })
        })
        .await?;

        Ok(())
    })
    .await;
}
