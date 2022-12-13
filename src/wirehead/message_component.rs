use super::{simulation::AsPhenotype, Session};
use crate::custom_id as cid;
use parking_lot::Mutex;
use serenity::{
    http::Http,
    model::prelude::{
        interaction::{message_component::MessageComponentInteraction, InteractionResponseType},
        ChannelId,
    },
    prelude::Mentionable,
};
use std::collections::HashMap;

pub async fn rate(
    sessions: &Mutex<HashMap<ChannelId, Session>>,
    http: &Http,
    mci: MessageComponentInteraction,
    id: super::TextGenome,
    rating: cid::Wirehead,
) -> anyhow::Result<()> {
    if !sessions.lock().contains_key(&mci.channel_id) {
        mci.create_interaction_response(http, |m| {
            m.kind(InteractionResponseType::ChannelMessageWithSource)
                .interaction_response_data(|d| d.content("there is no active session"))
        })
        .await?;

        return Ok(());
    };

    // this is a bit of a contortion but it's fine for now
    let (tags, hide_prompt) = if let Some(session) = sessions.lock().get(&mci.channel_id) {
        session.rate(
            id.clone(),
            match rating {
                cid::Wirehead::Negative2 => 0,
                cid::Wirehead::Negative1 => 25,
                cid::Wirehead::Zero => 50,
                cid::Wirehead::Positive1 => 75,
                cid::Wirehead::Positive2 => 100,
            },
        );
        (session.tags.clone(), session.hide_prompt)
    } else {
        (vec![], false)
    };

    mci.create_interaction_response(http, |m| {
        m.kind(InteractionResponseType::UpdateMessage)
            .interaction_response_data(|d| {
                d.content(format!(
                    "{}**Rating**: {} by {}",
                    if !hide_prompt {
                        format!("`{}`: | ", id.as_text(&tags))
                    } else {
                        String::new()
                    },
                    rating.as_integer(),
                    mci.user.mention(),
                ))
                .components(|c| c.set_action_rows(vec![]))
            })
    })
    .await?;

    Ok(())
}
